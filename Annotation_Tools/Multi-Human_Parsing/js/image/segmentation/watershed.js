/**
 * Canny + Watershed segmentation algorithm.
 *
 *  var segmentation = new WatershedSegmentation(imageData);
 *  var result = segmentation.result;
 *  var result = segmentation.finer();
 *  var result = segmentation.coarser();
 *
 *  TODO:
 *  * Edge options other than canny.
 *  * Create a graph-structure for coarse/fine adjustment.
 *
 */
define(["./base",
        "./binary-heap-priority-queue",
        "../canny",
        "../compat",
        "../distance-transform"],
function (BaseSegmentation, PriorityQueue, canny, compat, distanceTransform) {
  // Constructor for the segmentation configuration.
  function WatershedSegmentation(imageData, options) {
    BaseSegmentation.call(this, imageData, options);
    options = options || {};
    this.sigmaRange = options.sigmaRange ||
      [-2, -1, 0, 0.5, 1, 2, 3].map(function(n){
        return Math.pow(2, n);
      });
    this.kernelRange = options.kernelRange || [2, 3, 4, 4, 4, 5, 6];
    this.currentConfig = options.currentConfig ||
                         Math.floor((this.sigmaRange.length - 1) / 2);
    this.minRegionSize = options.minRegionSize || 16;
    this.highThreshold = options.highThreshold || 0.04;
    this.lowThreshold = options.lowThreshold || 0.3 * options.highThreshold;
    if (this.sigmaRange.length <= 0)
      throw "Invalid sigma range";
    this.neighborMap8 = new NeighborMap(this.imageData.width,
                                        this.imageData.height);
    this.neighborMap4 = new NeighborMap(this.imageData.width,
                                        this.imageData.height,
                                        [[-1, -1],
                                         [-1, 0],
                                         [-1, 1],
                                         [ 0, -1]]);
    this._compute();
  }

  WatershedSegmentation.prototype = Object.create(BaseSegmentation.prototype);

  // Change the segmentation resolution.
  WatershedSegmentation.prototype.finer = function () {
    if (this.currentConfig > 0) {
      --this.currentConfig;
      if (this.imageData)
        this._compute();
    }
  };

  // Change the segmentation resolution.
  WatershedSegmentation.prototype.coarser = function () {
    if (this.currentConfig < this.sigmaRange.length - 1) {
      ++this.currentConfig;
      if (this.imageData)
        this._compute();
    }
  };

  // Compute canny-watershed segmentation.
  WatershedSegmentation.prototype._compute = function () {
    var queue = new PriorityQueue({
      comparator: function(a, b) { return a[0] - b[0]; }
    });
    var edge = canny(this.imageData, {
      kernelTail: this.kernelRange[this.currentConfig],
      sigma: this.sigmaRange[this.currentConfig],
      lowThreshold: this.lowThreshold,
      highThreshold: this.highThreshold
    });
    var seeds = this._findLocalMaxima(distanceTransform(edge));
    var labels = new Int32Array(edge.data.length);
    var i, j, offset, neighbors, neighborOffset;
    // Initialize.
    for (i = 0; i < labels.length; ++i)
      labels[i] = -1;
    for (i = 0; i < seeds.length; ++i)
      labels[seeds[i]] = i + 1;
    for (i = 0; i < seeds.length; ++i) {
      neighbors = this.neighborMap8.get(seeds[i]);
      for (j = 0; j < neighbors.length; ++j) {
        neighborOffset = neighbors[j];
        if (labels[neighborOffset] === -1) {
          queue.push([edge.magnitude[neighborOffset], neighborOffset]);
          labels[neighborOffset] = -2;
        }
      }
    }
    // Iterate until we label all pixels by non-border dilation.
    var iter = 0;
    while (queue.length > 0) {
      offset = queue.shift()[1];
      neighbors = this.neighborMap8.get(offset);
      var uniqueLabel = this._findUniqueRegionLabel(neighbors, labels);
      if (uniqueLabel) {  // Dilate when there is a unique region label.
        labels[offset] = uniqueLabel;
        for (i = 0; i < neighbors.length; ++i) {
          neighborOffset = neighbors[i];
          if (labels[neighborOffset] === -1) {
            labels[neighborOffset] = -2;
            queue.push([edge.magnitude[neighborOffset], neighborOffset]);
          }
        }
      }
      else
        labels[offset] = 0;  // Boundary.
      if (++iter > labels.length)
        throw "Too many iterations";
    }
    // Remove boundaries and small regions.
    this.erode(0, labels);
    this._removeSmallRegions(labels);
    var numSegments = this._relabel(labels);
    this.result = this._encodeLabels(labels);
    this.result.numSegments = numSegments;
  };

  // Find the local maxima.
  WatershedSegmentation.prototype._findLocalMaxima = function (intensity) {
    var data = intensity.data,
        maximaMap = new Uint8Array(data.length),
        offsets = [],
        k, offset, neighbors, flag;
    for (offset = 0; offset < data.length; ++offset) {
      neighbors = this.neighborMap8.get(offset);
      flag = true;
      for (k = 0; k < neighbors.length; ++k)
        flag = flag && data[offset] >= data[neighbors[k]];
      maximaMap[offset] = flag;
    }
    // Erase connected seeds.
    var suppressed = new Uint8Array(maximaMap.length);
    for (offset = 0; offset < data.length; ++offset) {
      neighbors = this.neighborMap4.get(offset);
      flag = true;
      for (k = 0; k < neighbors.length; ++k)
        flag = flag && maximaMap[offset] > maximaMap[neighbors[k]];
      suppressed[offset] = flag;
    }
    for (offset = 0; offset < suppressed.length; ++offset)
      if (suppressed[offset])
        offsets.push(offset);
    return offsets;
  };

  WatershedSegmentation.prototype._findUniqueRegionLabel =
      function (neighbors, labels) {
    var uniqueLabels = [];
    for (var i = 0; i < neighbors.length; ++i) {
      var label = labels[neighbors[i]];
      if (label > 0 && uniqueLabels.indexOf(label) < 0)
        uniqueLabels.push(label);
    }
    return (uniqueLabels.length === 1) ? uniqueLabels[0] : null;
  };

  WatershedSegmentation.prototype._findDominantLabel =
      function (neighbors, labels, target) {
    var histogram = {},
        label;
    for (var i = 0; i < neighbors.length; ++i) {
      label = labels[neighbors[i]];
      if (label !== target) {
        if (histogram[label])
          ++histogram[label];
        else
          histogram[label] = 1;
      }
    }
    var count = 0,
        dominantLabel = null;
    for (label in histogram) {
      if (histogram[label] > count) {
        dominantLabel = label;
        count = histogram[label];
      }
    }
    return dominantLabel;
  };

  // Greedy erode.
  WatershedSegmentation.prototype.erode = function (target, labels) {
    var offsets = [],
        updates = {},
        offset;
    for (offset = 0; offset < labels.length; ++offset)
      if (labels[offset] === target)
        offsets.push(offset);
    if (target !== 0 && offsets.length === 0)
      throw "No pixels for label " + target;
    updates[target] = 0;
    var iter = 0;
    while (offsets.length > 0) {
      offset = offsets.shift();
      var neighbors = this.neighborMap8.get(offset),
          dominantLabel = this._findDominantLabel(neighbors, labels, target);
      if (dominantLabel !== null) {
        labels[offset] = dominantLabel;
        if (updates[dominantLabel])
          ++updates[dominantLabel];
        else
          updates[dominantLabel] = 1;
        --updates[target];
      }
      else
        offsets.push(offset);
      if (++iter > labels.length)
        throw "Too many iterations for label " + target;
    }
    return updates;
  };

  // Find small item.
  WatershedSegmentation.prototype._findSmallLabel =
      function (histogram) {
    var smallLabel = null;
    for (var label in histogram) {
      var count = histogram[label];
      if (0 < count && count < this.minRegionSize) {
        smallLabel = parseInt(label, 10);
        break;
      }
    }
    return smallLabel;
  };

  // Remove small regions.
  WatershedSegmentation.prototype._removeSmallRegions =
      function (labels) {
    var histogram = {},
        offset, label, updates;
    for (offset = 0; offset < labels.length; ++offset) {
      label = labels[offset];
      if (histogram[label])
        ++histogram[label];
      else
        histogram[label] = 1;
    }
    var iter = 0;
    while (true) {
      var smallLabel = this._findSmallLabel(histogram);
      if (smallLabel !== null) {
        updates = this.erode(smallLabel, labels);
        for (label in updates)
          histogram[label] += updates[label];
      }
      else
        break;
      if (++iter >= Object.keys(histogram).length)
        throw "Too many iterations";
    }
  };

  WatershedSegmentation.prototype._relabel = function (labels) {
    var uniqueArray = [];
    for (var i = 0; i < labels.length; ++i) {
      var index = uniqueArray.indexOf(labels[i]);
      if (index < 0) {
        index = uniqueArray.length;
        uniqueArray.push(labels[i]);
      }
      labels[i] = index;
    }
    return uniqueArray.length;
  };

  // Encode segmentation.
  WatershedSegmentation.prototype._encodeLabels = function (labels) {
    var imageData = new ImageData(this.imageData.width,
                                  this.imageData.height),
        data = imageData.data;
    for (var i = 0; i < labels.length; ++i) {
      var value = labels[i];
      data[4 * i] = 255 & value;
      data[4 * i + 1] = 255 & (value >> 8);
      data[4 * i + 2] = 255 & (value >> 16);
      data[4 * i + 3] = 255;
    }
    return imageData;
  };

  // Neighbor Map.
  function NeighborMap(width, height, neighbors) {
    this.neighbors = neighbors || [[-1, -1], [-1, 0], [-1, 1],
                                   [ 0, -1],          [ 0, 1],
                                   [ 1, -1], [ 1, 0], [ 1, 1]];
    this.maps = [];
    for (var k = 0; k < this.neighbors.length; ++k) {
      var dy = this.neighbors[k][0],
          dx = this.neighbors[k][1],
          map = new Int32Array(width * height);
      for (var y = 0; y < height; ++y) {
        for (var x = 0; x < width; ++x) {
          var Y = y + dy,
              X = x + dx;
          map[y * width + x] = (Y < 0 || height <= Y || X < 0 || width <= X) ?
                               -1 : Y * width + X;
        }
      }
      this.maps.push(map);
    }
  }

  NeighborMap.prototype.get = function (offset) {
    var neighborOffsets = [];
    for (var k = 0; k < this.neighbors.length; ++k) {
      var neighborOffset = this.maps[k][offset];
      if (neighborOffset >= 0)
        neighborOffsets.push(neighborOffset);
    }
    return neighborOffsets;
  };


  return WatershedSegmentation;
});
