/**
 * Base class for over-segmentation algorithms.
 *
 * Copyright 2015  Kota Yamaguchi
 */
define(["../compat"],
function (compat) {
  function BaseSegmentation(imageData, options) {
    if (!(imageData instanceof ImageData))
      throw "Invalid ImageData";
    this.imageData = compat.createImageData(imageData.width, imageData.height);
    this.imageData.data.set(imageData.data);
  }

  BaseSegmentation.prototype.finer = function () {};

  BaseSegmentation.prototype.coarser = function () {};

  return BaseSegmentation;
});
