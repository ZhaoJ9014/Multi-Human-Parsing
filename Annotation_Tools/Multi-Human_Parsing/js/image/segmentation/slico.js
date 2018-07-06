/** SLICO segmentation implementation.
 *
 *    SLIC Superpixels
 *    Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal
 *    Fua, and Sabine Süsstrunk
 *    IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34,
 *    num. 11, p. 2274 - 2282, May 2012.
 *
 *  http://ivrl.epfl.ch/research/superpixels
 *
 * Copyright 2015  Kota Yamaguchi
 */
define(["./base",
        "../compat"],
function(BaseSegmentation, compat) {
  function SLICO(imageData, options) {
    BaseSegmentation.call(this, imageData, options);
    this.width  = this.imageData.width;
    this.height = this.imageData.height;
    options = options || {};
    this.method = options.method || "FixedK";
    this.perturb = (typeof options.perturb === "undefined") ?
            true : options.perturb;
    this.maxIterations = options.maxIterations || 10;
    this.K = options.K || 1024;
    this.step = options.step || 200;
    this.enforceConnectivity = (options.enforceConnectivity === false) ?
                                false : true;
    this._compute();
  }

  SLICO.prototype = Object.create(BaseSegmentation.prototype);

  SLICO.prototype.finer = function () {
    var newK = Math.min(8962, Math.round(this.K * (2.0)));
    if (newK !== this.K) {
      this.K = newK;
      this._compute();
    }
  };

  SLICO.prototype.coarser = function () {
    var newK = Math.max(16, Math.round(this.K / (2.0)));
    if (newK !== this.K) {
      this.K = newK;
      this._compute();
    }
  };

  SLICO.prototype._compute = function () {
    var labels = (this.method === "FixedK") ?
        this.performSLICOForGivenK() : this.performSLICOForGivenStepSize();
    var result = new ImageData(this.width, this.height);
    result.numSegments = remapLabels(labels);
    encodeLabels(labels, result.data);
    this.result = result;
  };

  // sRGB (D65 illuninant assumption) to XYZ conversion.
  SLICO.prototype.rgb2xyz = function (sRGB) {
    var R = parseInt(sRGB[0], 10) / 255.0,
        G = parseInt(sRGB[1], 10) / 255.0,
        B = parseInt(sRGB[2], 10) / 255.0,
        r = (R <= 0.04045) ? R / 12.92 : Math.pow((R + 0.055) / 1.055, 2.4),
        g = (G <= 0.04045) ? G / 12.92 : Math.pow((R + 0.055) / 1.055, 2.4),
        b = (B <= 0.04045) ? B / 12.92 : Math.pow((R + 0.055) / 1.055, 2.4);
    return [
      r * 0.4124564 + g * 0.3575761 + b * 0.1804375,
      r * 0.2126729 + g * 0.7151522 + b * 0.0721750,
      r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    ];
  };

  // sRGB to Lab conversion.
  SLICO.prototype.rgb2lab = function (sRGB) {
    var epsilon = 0.008856,  //actual CIE standard
        kappa   = 903.3,     //actual CIE standard
        Xr = 0.950456,       //reference white
        Yr = 1.0,            //reference white
        Zr = 1.088754,       //reference white
        xyz = this.rgb2xyz(sRGB),
        xr = xyz[0] / Xr,
        yr = xyz[1] / Yr,
        zr = xyz[2] / Zr,
        fx = (xr > epsilon) ?
            Math.pow(xr, 1.0/3.0) : (kappa * xr + 16.0) / 116.0,
        fy = (yr > epsilon) ?
            Math.pow(yr, 1.0/3.0) : (kappa * yr + 16.0) / 116.0,
        fz = (zr > epsilon) ?
            Math.pow(zr, 1.0/3.0) : (kappa * zr + 16.0) / 116.0;
    return [
      116.0 * fy - 16.0,
      500.0 * (fx - fy),
      200.0 * (fy - fz)
    ];
  };

  SLICO.prototype.doRGBtoLABConversion = function (imageData) {
    var size = this.width * this.height,
        data = imageData.data;
    this.lvec = new Float64Array(size);
    this.avec = new Float64Array(size);
    this.bvec = new Float64Array(size);
    for (var j = 0; j < size; ++j) {
      var r = data[4 * j + 0],
          g = data[4 * j + 1],
          b = data[4 * j + 2];
      var lab = this.rgb2lab([r, g, b]);
      this.lvec[j] = lab[0];
      this.avec[j] = lab[1];
      this.bvec[j] = lab[2];
    }
  };

  SLICO.prototype.detectLabEdges = function () {
    var w = this.width;
    this.edges = fillArray(new Float64Array(this.width * this.height), 0);
    for (var j = 1; j < this.height - 1; ++j) {
      for (var k = 1; k < this.width - 1; ++k) {
        var i = parseInt(j * this.width + k, 10),
            dx = Math.pow(this.lvec[i - 1] - this.lvec[i + 1], 2) +
                 Math.pow(this.avec[i - 1] - this.avec[i + 1], 2) +
                 Math.pow(this.bvec[i - 1] - this.bvec[i + 1], 2),
            dy = Math.pow(this.lvec[i - w] - this.lvec[i + w], 2) +
                 Math.pow(this.avec[i - w] - this.avec[i + w], 2) +
                 Math.pow(this.bvec[i - w] - this.bvec[i + w], 2);
        this.edges[i] = dx + dy;
      }
    }
  };

  SLICO.prototype.perturbSeeds = function () {
    var dx8 = [-1, -1,  0,  1, 1, 1, 0, -1],
        dy8 = [ 0, -1, -1, -1, 0, 1, 1,  1],
        numSeeds = this.kSeedsL.length;
    for (var n = 0; n < numSeeds; ++n) {
      var ox = parseInt(this.kSeedsX[n], 10),  //original x
          oy = parseInt(this.kSeedsY[n], 10),  //original y
          oind = parseInt(oy * this.width + ox, 10),
          storeind = parseInt(oind, 10);
      for (var i = 0; i < 8; ++i) {
        var nx = parseInt(ox + dx8[i], 10);  //new x
        var ny = parseInt(oy + dy8[i], 10);  //new y
        if (nx >= 0 && nx < this.width && ny >= 0 && ny < this.height) {
          var nind = parseInt(ny * this.width + nx, 10);
          if (this.edges[nind] < this.edges[storeind])
            storeind = nind;
        }
      }
      if (storeind != oind) {
        this.kSeedsX[n] = Math.floor(storeind % this.width);
        this.kSeedsY[n] = Math.floor(storeind / this.width);
        this.kSeedsL[n] = this.lvec[storeind];
        this.kSeedsA[n] = this.avec[storeind];
        this.kSeedsB[n] = this.bvec[storeind];
      }
    }
  };

  SLICO.prototype.getLABXYSeedsForGivenStepSize = function(step, perturb) {
    var n = 0,
        xstrips = Math.round(0.5 + parseFloat(this.width) / parseFloat(step)),
        ystrips = Math.round(0.5 + parseFloat(this.height) / parseFloat(step)),
        xerr = Math.round(this.width  - step * xstrips),
        yerr = Math.round(this.height - step * ystrips),
        xerrperstrip = parseFloat(xerr) / parseFloat(xstrips),
        yerrperstrip = parseFloat(yerr) / parseFloat(ystrips),
        xoff = Math.floor(step / 2),
        yoff = Math.floor(step / 2),
        numSeeds = xstrips * ystrips;
    this.kSeedsL = new Float64Array(numSeeds);
    this.kSeedsA = new Float64Array(numSeeds);
    this.kSeedsB = new Float64Array(numSeeds);
    this.kSeedsX = new Float64Array(numSeeds);
    this.kSeedsY = new Float64Array(numSeeds);
    for (var y = 0; y < ystrips; ++y) {
      var ye = Math.floor(y * yerrperstrip);
      for (var x = 0; x < xstrips; ++x) {
        var xe = Math.floor(x * xerrperstrip);
        var i = Math.floor((y * step + yoff + ye) * this.width +
                           (x * step + xoff + xe));
        this.kSeedsL[n] = this.lvec[i];
        this.kSeedsA[n] = this.avec[i];
        this.kSeedsB[n] = this.bvec[i];
        this.kSeedsX[n] = (x * step + xoff + xe);
        this.kSeedsY[n] = (y * step + yoff + ye);
        ++n;
      }
    }
    if (perturb)
      this.perturbSeeds();
  };

  SLICO.prototype.getLABXYSeedsForGivenK = function(K, perturb) {
    var size = Math.floor(this.width * this.height);
    var step = Math.sqrt(parseFloat(size) / parseFloat(K));
    var xoff = Math.round(step / 2);
    var yoff = Math.round(step / 2);
    var n = 0;
    var r = 0;
    this.kSeedsL = [];
    this.kSeedsA = [];
    this.kSeedsB = [];
    this.kSeedsX = [];
    this.kSeedsY = [];
    for (var y = 0; y < this.height; ++y) {
      var Y = Math.floor(y * step + yoff);
      if (Y > this.height - 1)
        break;
      for (var x = 0; x < this.width; ++x) {
        //var X = x*step + xoff;  //square grid
        var X = Math.floor(x * step + (xoff << (r & 0x1)));  //hex grid
        if (X > this.width - 1)
          break;
        var i = Math.floor(Y * this.width + X);
        this.kSeedsL.push(this.lvec[i]);
        this.kSeedsA.push(this.avec[i]);
        this.kSeedsB.push(this.bvec[i]);
        this.kSeedsX.push(X);
        this.kSeedsY.push(Y);
        ++n;
      }
      ++r;
    }
    if (perturb)
      this.perturbSeeds();
  };

  function fillArray(array, value) {
    for (var i = 0; i < array.length; ++i)
      array[i] = value;
    return array;
  }

  // function findMinMax(data) {
  //   var min = Infinity, max = -Infinity;
  //   for (var i = 0; i < data.length; ++i) {
  //     min = Math.min(min, data[i]);
  //     max = Math.max(max, data[i]);
  //   }
  //   return [min, max];
  // }

  // function sum(data) {
  //   var value = 0;
  //   for (var i = 0; i < data.length; ++i)
  //     value += data[i];
  //   return value;
  // }

  SLICO.prototype.performSuperpixelSegmentationVariableSandM = function (
    kLabels,
    step,
    maxIterations
    ) {
    var size = Math.floor(this.width * this.height),
        numK = this.kSeedsL.length,
        numIter = 0,
        offset = Math.floor((step < 10) ? step * 1.5 : step),
        sigmal = fillArray(new Float64Array(numK), 0),
        sigmaa = fillArray(new Float64Array(numK), 0),
        sigmab = fillArray(new Float64Array(numK), 0),
        sigmax = fillArray(new Float64Array(numK), 0),
        sigmay = fillArray(new Float64Array(numK), 0),
        clusterSize = fillArray(new Int32Array(numK), 0),
        distxy = fillArray(new Float64Array(size), Infinity),
        distlab = fillArray(new Float64Array(size), Infinity),
        distvec = fillArray(new Float64Array(size), Infinity),
        maxlab = fillArray(new Float64Array(numK), Math.pow(10, 2)),
        maxxy = fillArray(new Float64Array(numK), Math.pow(step, 2)),
        i, j, k, n, x, y;
    while (numIter < maxIterations) {
      ++numIter;
      // Assign the closest cluster.
      fillArray(distvec, Infinity);
      for (n = 0; n < numK; ++n) {
        var y1 = Math.floor(Math.max(0, this.kSeedsY[n] - offset)),
            y2 = Math.floor(Math.min(this.height, this.kSeedsY[n] + offset)),
            x1 = Math.floor(Math.max(0, this.kSeedsX[n] - offset)),
            x2 = Math.floor(Math.min(this.width, this.kSeedsX[n] + offset));
        for (y = y1; y < y2; ++y) {
          for (x = x1; x < x2; ++x) {
            i = Math.floor(y * this.width + x);
            if (!(y < this.height && x < this.width && y >= 0 && x >= 0))
              throw "Assertion error";
            var l = this.lvec[i],
                a = this.avec[i],
                b = this.bvec[i];
            distlab[i] = Math.pow(l - this.kSeedsL[n], 2) +
                         Math.pow(a - this.kSeedsA[n], 2) +
                         Math.pow(b - this.kSeedsB[n], 2);
            distxy[i] = Math.pow(x - this.kSeedsX[n], 2) +
                        Math.pow(y - this.kSeedsY[n], 2);
            var dist = distlab[i] / maxlab[n] + distxy[i] / maxxy[n];
            if (dist < distvec[i]) {
              distvec[i] = dist;
              kLabels[i] = n;
            }
          }
        }
      }
      //console.log("iter = " + numIter + ", sum_dist = " + sum(distvec));
      // Assign the max color distance for a cluster.
      if (numIter === 0) {
        fillArray(maxlab, 1);
        fillArray(maxxy, 1);
      }
      for (i = 0; i < size; ++i) {
        if (maxlab[kLabels[i]] < distlab[i])
          maxlab[kLabels[i]] = distlab[i];
        if (maxxy[kLabels[i]] < distxy[i])
          maxxy[kLabels[i]] = distxy[i];
      }
      // Recalculate the centroid and store in the seed values.
      fillArray(sigmal, 0);
      fillArray(sigmaa, 0);
      fillArray(sigmab, 0);
      fillArray(sigmax, 0);
      fillArray(sigmay, 0);
      fillArray(clusterSize, 0);
      for (j = 0; j < size; ++j) {
        var temp = kLabels[j];
        if (temp < 0)
          throw "Assertion error";
        sigmal[temp] += this.lvec[j];
        sigmaa[temp] += this.avec[j];
        sigmab[temp] += this.bvec[j];
        sigmax[temp] += (j % this.width);
        sigmay[temp] += (j / this.width);
        clusterSize[temp]++;
      }
      for (k = 0; k < numK; ++k) {
        if (clusterSize[k] <= 0)
          clusterSize[k] = 1;
        //computing inverse now to multiply, than divide later.
        var inv = 1.0 / clusterSize[k];
        this.kSeedsL[k] = sigmal[k] * inv;
        this.kSeedsA[k] = sigmaa[k] * inv;
        this.kSeedsB[k] = sigmab[k] * inv;
        this.kSeedsX[k] = sigmax[k] * inv;
        this.kSeedsY[k] = sigmay[k] * inv;
      }
    }
  };

  SLICO.prototype.enforceLabelConnectivity = function (labels, nlabels, K) {
    var dx4 = [-1,  0,  1,  0],
        dy4 = [ 0, -1,  0,  1],
        size = this.width * this.height,
        SUPSZ = Math.floor(size / K),
        c, n, x, y, nindex;
    var label = 0,
        xvec = new Int32Array(size),
        yvec = new Int32Array(size),
        oindex = 0,
        adjlabel = 0;  // adjacent label
    for (var j = 0; j < this.height; ++j) {
      for (var k = 0; k < this.width; ++k) {
        if (nlabels[oindex] < 0) {
          nlabels[oindex] = label;
          // Start a new segment.
          xvec[0] = k;
          yvec[0] = j;
          //  Quickly find an adjacent label for use later if needed.
          for (n = 0; n < 4; ++n) {
            x = Math.floor(xvec[0] + dx4[n]);
            y = Math.floor(yvec[0] + dy4[n]);
            if ((x >= 0 && x < this.width) && (y >= 0 && y < this.height)) {
              nindex = Math.floor(y * this.width + x);
              if (nlabels[nindex] >= 0)
                adjlabel = nlabels[nindex];
            }
          }
          var count = 1;
          for (c = 0; c < count; ++c) {
            for (n = 0; n < 4; ++n) {
              x = Math.floor(xvec[c] + dx4[n]);
              y = Math.floor(yvec[c] + dy4[n]);
              if ((x >= 0 && x < this.width) && (y >= 0 && y < this.height)) {
                nindex = Math.floor(y * this.width + x);
                if (nlabels[nindex] < 0 && labels[oindex] == labels[nindex]) {
                  xvec[count] = x;
                  yvec[count] = y;
                  nlabels[nindex] = label;
                  ++count;
                }
              }
            }
          }
          // If segment size is less then a limit, assign an
          // adjacent label found before, and decrement label count.
          if (count <= SUPSZ >> 2) {
            for (c = 0; c < count; c++ ) {
              var ind = Math.floor(yvec[c] * this.width + xvec[c]);
              nlabels[ind] = adjlabel;
            }
            --label;
          }
          ++label;
        }
        ++oindex;
      }
    }
    return label;
  };

  SLICO.prototype.performSLICOForGivenStepSize = function() {
    var size = this.width * this.height,
        kLabels = fillArray(new Int32Array(size), -1);
    this.doRGBtoLABConversion(this.imageData);
    if (this.perturb)
      this.detectLabEdges();
    this.getLABXYSeedsForGivenStepSize(this.step, this.perturb);
    this.performSuperpixelSegmentationVariableSandM(kLabels,
                                                    this.step,
                                                    this.maxIterations);
    var numlabels = kLabels.length;
    if (this.enforceConnectivity) {
      var nlabels = fillArray(new Int32Array(size), -1);
      numlabels = this.enforceLabelConnectivity(kLabels,
                                                nlabels,
                                                size / (this.step * this.estep));
      for (var i = 0; i < size; ++i)
        kLabels[i] = nlabels[i];
    }
    return kLabels;
  };

  SLICO.prototype.performSLICOForGivenK = function() {
    var size = this.width * this.height,
        kLabels = fillArray(new Int32Array(size), -1);
    this.doRGBtoLABConversion(this.imageData);
    if (this.perturb)
      this.detectLabEdges();
    this.getLABXYSeedsForGivenK(this.K, this.perturb);
    var step = Math.sqrt(size / this.K) + 2.0;
    this.performSuperpixelSegmentationVariableSandM(kLabels,
                                                    step,
                                                    this.maxIterations);
    var numlabels = kLabels.length;
    if (this.enforceConnectivity) {
      var nlabels = fillArray(new Int32Array(size), -1);
      numlabels = this.enforceLabelConnectivity(kLabels, nlabels, this.K);
      for (var i = 0; i < size; ++i)
        kLabels[i] = nlabels[i];
    }
    return kLabels;
  };

  SLICO.prototype.drawContoursAroundSegments = function (result) {
    var imageData = new ImageData(this.width, this.height),
        data = fillArray(imageData.data, 255),
        color = [255, 0, 0],
        dx8 = [-1, -1,  0,  1, 1, 1, 0, -1],
        dy8 = [ 0, -1, -1, -1, 0, 1, 1,  1],
        istaken = fillArray(new Uint8Array(this.width * this.height), 0);
    var mainindex = 0;
    for (var j = 0; j < this.height; ++j) {
      for (var k = 0; k < this.width; ++k) {
        var np = 0;
        for (var i = 0; i < 8; ++i) {
          var x = k + dx8[i],
              y = j + dy8[i];
          if ((x >= 0 && x < this.width) && (y >= 0 && y < this.height)) {
            var index = y * this.width + x;
            if (istaken[index] === 0 &&
                result.labels[mainindex] !== result.labels[index])
              ++np;
          }
        }
        if (np > 1) {
          data[4 * mainindex + 0] = color[0];
          data[4 * mainindex + 1] = color[1];
          data[4 * mainindex + 2] = color[2];
        }
        ++mainindex;
      }
    }
    return imageData;
  };

  // Remap label indices.
  function remapLabels(labels) {
    var map = {},
        index = 0;
    for (var i = 0; i < labels.length; ++i) {
      var label = labels[i];
      if (map[label] === undefined)
        map[label] = index++;
        labels[i] = map[label];
    }
    return index;
  }

  function encodeLabels(labels, data) {
    for (var i = 0; i < labels.length; ++i) {
      var label = labels[i];
      data[4 * i + 0] = 255 & label;
      data[4 * i + 1] = 255 & (label >> 8);
      data[4 * i + 2] = 255 & (label >> 16);
      data[4 * i + 3] = 255;
    }
  }

  return SLICO;
});
