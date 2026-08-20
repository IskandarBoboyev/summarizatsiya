/**
 * macOS Vision OCR (JXA). Xcode / Tesseract shart emas.
 * Yo'l: OCR_IMAGE_PATH muhit o'zgaruvchisi.
 */
ObjC.import("Vision");
ObjC.import("AppKit");
ObjC.import("Foundation");

function ocrPath(path) {
  const url = $.NSURL.fileURLWithPath(path);
  const img = $.NSImage.alloc.initWithContentsOfURL(url);
  if (!img) {
    return "";
  }
  const tiff = img.TIFFRepresentation;
  const bmp = $.NSBitmapImageRep.alloc.initWithData(tiff);
  const cg = bmp.CGImage;
  if (!cg) {
    return "";
  }

  const request = $.VNRecognizeTextRequest.alloc.init;
  request.recognitionLevel = $.VNRequestTextRecognitionLevelAccurate;
  request.usesLanguageCorrection = true;
  try {
    request.recognitionLanguages = ["ru-RU", "en-US", "uz", "tg"];
  } catch (e) {
    /* til ro'yxati ixtiyoriy */
  }

  const handler = $.VNImageRequestHandler.alloc.initWithCGImageOptions(cg, {});
  const err = $();
  const ok = handler.performRequestsError($([request]), err);
  if (!ok) {
    return "";
  }
  const results = request.results;
  if (!results) {
    return "";
  }
  const lines = [];
  for (let i = 0; i < results.count; i++) {
    const cands = results.objectAtIndex(i).topCandidates(1);
    if (cands && cands.count > 0) {
      lines.push(ObjC.unwrap(cands.objectAtIndex(0).string));
    }
  }
  return lines.join("\n");
}

const envPath = ObjC.unwrap(
  $.NSProcessInfo.processInfo.environment.objectForKey("OCR_IMAGE_PATH")
);
function writeStdout(text) {
  const data = $.NSString.alloc.initWithString(text).dataUsingEncoding($.NSUTF8StringEncoding);
  $.NSFileHandle.fileHandleWithStandardOutput.writeData(data);
}

if (envPath) {
  const text = ocrPath(envPath);
  if (text) {
    writeStdout(text);
  }
}
