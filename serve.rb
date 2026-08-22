#!/usr/bin/env ruby
# frozen_string_literal: true

# FastAPI o'rniga vaqtinchalik WEBrick server.
# Sabab: ushbu Mac da Xcode CLI / ishlaydigan Python yo'q.
# UI va API shu yerda ishlaydi; og'ir LLM/ASR keyin Python o'rnatilgach qo'shiladi.

require "webrick"
require "json"
require "fileutils"
require "time"
require "cgi"
require "open3"

ROOT = File.expand_path(__dir__)
FRONTEND = File.join(ROOT, "frontend")
STATIC = File.join(FRONTEND, "static")
TEMPLATES = File.join(FRONTEND, "templates")
DATA_DIR = File.join(ROOT, "data")
WATCHED = File.join(ROOT, "watched_folders")
DB_PATH = File.join(DATA_DIR, "documents.db")
SQLITE = "/usr/bin/sqlite3"

FileUtils.mkdir_p(DATA_DIR)
FileUtils.mkdir_p(WATCHED)

def utc_now
  Time.now.utc.iso8601
end

def sql(query, *params)
  # Parametrlarni oddiy escape (bir tirnoq)
  bound = query.dup
  params.each do |p|
    val = p.nil? ? "" : p.to_s.gsub("'", "''")
    bound.sub!("?", "'#{val}'")
  end
  out, err, st = Open3.capture3(SQLITE, "-json", DB_PATH, bound)
  raise "SQLite: #{err}" unless st.success?
  return [] if out.strip.empty?
  JSON.parse(out)
end

def sql_exec(query, *params)
  sql(query, *params)
  true
end

def init_db
  schema = <<~SQL
    CREATE TABLE IF NOT EXISTS watched_folders (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      path TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      filename TEXT NOT NULL,
      filepath TEXT NOT NULL UNIQUE,
      file_type TEXT NOT NULL,
      file_size INTEGER NOT NULL DEFAULT 0,
      mtime REAL NOT NULL DEFAULT 0,
      original_text TEXT NOT NULL DEFAULT '',
      summary TEXT NOT NULL DEFAULT '',
      translation_uz TEXT NOT NULL DEFAULT '',
      transcription TEXT NOT NULL DEFAULT '',
      status TEXT NOT NULL DEFAULT 'pending',
      error_message TEXT NOT NULL DEFAULT '',
      model_used TEXT NOT NULL DEFAULT '',
      asr_language TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
  SQL
  Open3.capture3(SQLITE, DB_PATH, schema)
  sql_exec("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", "llm_model", "gemma-4")
  sql_exec("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", "asr_language", "uz")
  sql_exec("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", "quantization", "auto")
  sql_exec("INSERT OR IGNORE INTO watched_folders (path, created_at) VALUES (?, ?)", File.expand_path(WATCHED), utc_now)
end

def settings_hash
  rows = sql("SELECT key, value FROM settings")
  rows.each_with_object({}) { |r, h| h[r["key"]] = r["value"] }
end

def queue_counts
  rows = sql("SELECT status, COUNT(*) AS c FROM documents GROUP BY status")
  counts = Hash.new(0)
  rows.each { |r| counts[r["status"]] = r["c"].to_i }
  total = sql("SELECT COUNT(*) AS c FROM documents").dig(0, "c").to_i
  { "pending" => counts["pending"], "processing" => counts["processing"], "done" => counts["done"], "error" => counts["error"], "total" => total }
end

def public_doc(row)
  return nil unless row
  row
end

def file_kind(path)
  ext = File.extname(path).downcase
  return "pdf" if ext == ".pdf"
  return "docx" if ext == ".docx"
  return "text" if %w[.txt .md .rtf].include?(ext)
  return "image" if %w[.png .jpg .jpeg .webp .tif .tiff .bmp].include?(ext)
  return "audio" if %w[.wav .mp3 .ogg .flac .m4a .aac .wma].include?(ext)
  return "video" if %w[.mp4 .mkv .mov .webm .avi].include?(ext)
  "other"
end

SUPPORTED = %w[.pdf .docx .txt .md .rtf .png .jpg .jpeg .webp .tif .tiff .bmp .wav .mp3 .ogg .flac .m4a .aac .wma .mp4 .mkv .mov .webm .avi]

def candidate?(path)
  name = File.basename(path)
  return false if name.start_with?(".", "~$", ".~")
  return false if name.end_with?(".tmp", ".part", ".crdownload", ".download")
  SUPPORTED.include?(File.extname(path).downcase)
end

def extract_text(path)
  ext = File.extname(path).downcase
  if %w[.txt .md .rtf].include?(ext)
    File.read(path, encoding: "UTF-8", invalid: :replace, undef: :replace)
  else
    nil
  end
end

def enqueue_file(path)
  return unless File.file?(path) && candidate?(path)
  path = File.expand_path(path)
  stat = File.stat(path)
  kind = file_kind(path)
  now = utc_now
  existing = sql("SELECT * FROM documents WHERE filepath = ?", path).first
  if existing && existing["status"] == "done" && existing["file_size"].to_i == stat.size
    return
  end
  if existing
    sql_exec(
      "UPDATE documents SET filename = ?, file_type = ?, file_size = ?, mtime = ?, status = 'pending', error_message = '', updated_at = ? WHERE filepath = ?",
      File.basename(path), kind, stat.size, stat.mtime.to_f, now, path
    )
  else
    sql_exec(
      "INSERT INTO documents (filename, filepath, file_type, file_size, mtime, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
      File.basename(path), path, kind, stat.size, stat.mtime.to_f, now, now
    )
  end
end

def scan_all
  folders = sql("SELECT path FROM watched_folders")
  folders.each do |f|
    dir = f["path"]
    next unless Dir.exist?(dir)
    Dir.glob(File.join(dir, "**", "*"), File::FNM_DOTMATCH).each do |p|
      next unless File.file?(p)
      enqueue_file(p)
    end
  end
end

def process_pending
  pending = sql("SELECT * FROM documents WHERE status = 'pending' ORDER BY id ASC LIMIT 1").first
  return unless pending
  id = pending["id"]
  path = pending["filepath"]
  now = utc_now
  sql_exec("UPDATE documents SET status = 'processing', updated_at = ? WHERE id = ?", now, id)
  begin
    unless File.file?(path)
      raise "Fayl yo'qolgan: #{path}"
    end
    text = extract_text(path)
    if text && !text.strip.empty?
      sql_exec(
        "UPDATE documents SET original_text = ?, summary = ?, translation_uz = ?, status = 'done', error_message = ?, updated_at = ? WHERE id = ?",
        text,
        text.strip.split(/\n+/).first(6).join("\n"),
        text,
        "",
        now,
        id
      )
    else
      sql_exec(
        "UPDATE documents SET status = 'error', error_message = ?, updated_at = ? WHERE id = ?",
        "Hozircha to'liq OCR/LLM/ASR ishlamayapti: bu Mac da Python (Xcode Command Line Tools) o'rnatilmagan. " \
        "Terminalda `xcode-select --install` qiling, keyin README bo'yicha .venv yarating. " \
        "TXT/MD fayllar shu serverda ham o'qiladi.",
        now,
        id
      )
    end
  rescue StandardError => e
    sql_exec("UPDATE documents SET status = 'error', error_message = ?, updated_at = ? WHERE id = ?", e.message, now, id)
  end
end

def json_res(res, data, status = 200)
  body = JSON.generate(data)
  res.status = status
  res["Content-Type"] = "application/json; charset=utf-8"
  res["Content-Length"] = body.bytesize.to_s
  res.body = body
end

def read_json(req)
  raw = req.body
  return {} if raw.nil? || raw.empty?
  JSON.parse(raw)
rescue JSON::ParserError
  {}
end

def query_params(req)
  CGI.parse(req.query_string || "").transform_values { |v| v.first }
end

def browse_dir(raw)
  home = File.expand_path("~")
  target = raw.to_s.strip.empty? ? home : File.expand_path(raw)
  unless File.directory?(target)
    raise "Papka topilmadi: #{target}"
  end

  entries = []
  begin
    Dir.children(target).sort_by { |n| n.downcase }.each do |name|
      next if name.start_with?(".")
      full = File.join(target, name)
      next unless File.directory?(full)
      entries << { "name" => name, "path" => full, "is_dir" => true }
    end
  rescue Errno::EACCES
    raise "Papka ochilmadi (ruxsat yo'q): #{target}"
  end

  parent = File.dirname(target)
  parent = nil if parent == target

  shortcuts = []
  [
    ["Uy", home],
    ["Ish stoli", File.join(home, "Desktop")],
    ["Hujjatlar", File.join(home, "Documents")],
    ["Yuklamalar", File.join(home, "Downloads")],
    ["Kuzatuv", File.expand_path(WATCHED)],
    ["Disk", "/"]
  ].each do |label, pth|
    next unless File.directory?(pth)
    shortcuts << { "name" => label, "path" => File.expand_path(pth) }
  end

  { "path" => target, "parent" => parent, "entries" => entries, "shortcuts" => shortcuts }
end

init_db
scan_all

# Fon: skaner + oddiy qayta ishlash
Thread.new do
  loop do
    begin
      scan_all
      process_pending
    rescue StandardError => e
      warn "worker: #{e}"
    end
    sleep 2
  end
end

server = WEBrick::HTTPServer.new(
  BindAddress: "127.0.0.1",
  Port: 8000,
  AccessLog: [],
  Logger: WEBrick::Log.new($stderr, WEBrick::Log::INFO)
)

trap("INT") { server.shutdown }
trap("TERM") { server.shutdown }

server.mount_proc "/" do |req, res|
  path = req.path
  method = req.request_method

  if path == "/" && method == "GET"
    html = File.read(File.join(TEMPLATES, "index.html"), encoding: "UTF-8")
    res["Content-Type"] = "text/html; charset=utf-8"
    res.body = html
    next
  end

  if path == "/gerb.mp4" && method == "GET"
    file = File.join(ROOT, "public", "gerb.mp4")
    unless File.file?(file)
      res.status = 404
      res.body = "Not found"
      next
    end
    res["Content-Type"] = "video/mp4"
    res["Cache-Control"] = "no-cache, must-revalidate"
    res.body = File.binread(file)
    next
  end

  if path.start_with?("/static/") && method == "GET"
    rel = path.sub(%r{^/static/}, "")
    file = File.expand_path(File.join(STATIC, rel))
    unless file.start_with?(File.expand_path(STATIC)) && File.file?(file)
      res.status = 404
      res.body = "Not found"
      next
    end
    ext = File.extname(file)
    res["Content-Type"] =
      case ext
      when ".css" then "text/css; charset=utf-8"
      when ".js" then "application/javascript; charset=utf-8"
      when ".png" then "image/png"
      when ".svg" then "image/svg+xml"
      else "application/octet-stream"
      end
    res.body = File.binread(file)
    next
  end

  if path == "/api/status"
    json_res(res, {
      "device" => { "device" => "cpu", "name" => "Ruby fallback (Python yo'q)", "memory_gb" => 0, "offline" => true, "recommended_quantization" => "none" },
      "queue" => queue_counts,
      "llm" => { "ready" => false, "model_key" => nil, "backend" => nil },
      "asr" => { "ready" => false, "backend" => nil, "error" => "Python o'rnatilmagan" },
      "models" => [
        { "key" => "gemma-4", "label" => "Gemma 4", "backend" => "missing", "path" => "models/gemma-4", "ready" => false },
        { "key" => "gemma-26", "label" => "Gemma 26", "backend" => "missing", "path" => "models/gemma-26", "ready" => false }
      ],
      "asr_languages" => {
        "auto" => "Avtomatik aniqlash", "uz" => "O'zbekcha", "ru" => "Ruscha", "en" => "Inglizcha",
        "kk" => "Qozoqcha", "ky" => "Qirg'izcha", "tg" => "Tojikcha", "tr" => "Turkcha"
      },
      "settings" => settings_hash,
      "offline" => true
    })
    next
  end

  if path == "/api/settings" && method == "GET"
    json_res(res, settings_hash)
    next
  end

  if path == "/api/settings" && method == "PUT"
    body = read_json(req)
    %w[llm_model asr_language quantization].each do |k|
      sql_exec("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", k, body[k]) if body[k]
    end
    json_res(res, settings_hash)
    next
  end

  if path == "/api/browse" && method == "GET"
    begin
      json_res(res, browse_dir(query_params(req)["path"]))
    rescue StandardError => e
      json_res(res, { "detail" => e.message }, 400)
    end
    next
  end

  if path == "/api/folders" && method == "GET"
    json_res(res, sql("SELECT id, path, created_at FROM watched_folders ORDER BY id ASC"))
    next
  end

  if path == "/api/folders" && method == "POST"
    body = read_json(req)
    folder = File.expand_path(body["path"].to_s)
    unless File.directory?(folder)
      json_res(res, { "detail" => "Papka topilmadi: #{folder}" }, 400)
      next
    end
    begin
      sql_exec("INSERT INTO watched_folders (path, created_at) VALUES (?, ?)", folder, utc_now)
      row = sql("SELECT id, path, created_at FROM watched_folders WHERE path = ?", folder).first
      scan_all
      json_res(res, row)
    rescue StandardError => e
      json_res(res, { "detail" => e.message }, 400)
    end
    next
  end

  if path =~ %r{^/api/folders/(\d+)$} && method == "DELETE"
    id = Regexp.last_match(1)
    sql_exec("DELETE FROM watched_folders WHERE id = ?", id)
    json_res(res, { "ok" => true })
    next
  end

  if path == "/api/scan" && method == "POST"
    scan_all
    json_res(res, { "enqueued" => 0, "queue" => queue_counts })
    next
  end

  if path == "/api/documents" && method == "GET"
    q = query_params(req)
    offset = q["offset"].to_i
    limit = (q["limit"] || "12").to_i
    limit = 12 if limit <= 0
    total = sql("SELECT COUNT(*) AS c FROM documents").dig(0, "c").to_i
    items = sql("SELECT * FROM documents ORDER BY id ASC LIMIT #{limit.to_i} OFFSET #{offset.to_i}")
    json_res(res, { "items" => items, "total" => total, "offset" => offset, "limit" => limit, "has_more" => offset + items.size < total })
    next
  end

  if path =~ %r{^/api/documents/(\d+)$} && method == "GET"
    row = sql("SELECT * FROM documents WHERE id = ?", Regexp.last_match(1)).first
    if row
      json_res(res, row)
    else
      json_res(res, { "detail" => "Hujjat topilmadi" }, 404)
    end
    next
  end

  if path =~ %r{^/api/documents/(\d+)/reprocess$} && method == "POST"
    id = Regexp.last_match(1)
    sql_exec("UPDATE documents SET status = 'pending', error_message = '', updated_at = ? WHERE id = ?", utc_now, id)
    row = sql("SELECT * FROM documents WHERE id = ?", id).first
    json_res(res, row || { "detail" => "Hujjat topilmadi" }, row ? 200 : 404)
    next
  end

  res.status = 404
  res["Content-Type"] = "application/json"
  res.body = JSON.generate({ "detail" => "Not found" })
end

$stderr.puts "UI: http://127.0.0.1:8000  (Ruby fallback — Python o'rnatilmagan)"
server.start
