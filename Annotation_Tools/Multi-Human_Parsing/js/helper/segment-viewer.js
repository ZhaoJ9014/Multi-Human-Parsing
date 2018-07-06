/** Segmentation viewer.
 *
 * var viewer = new Viewer("/path/to/image.jpg", "/path/to/annotation.png", {
 *   colormap: [[255, 255, 255], [0, 255, 255]],
 *   labels: ["background", "foreground"],
 *   onload: function () { }
 * });
 * document.body.appendChild(viewer.container);
 *
 * Copyright 2015  Kota Yamaguchi
 */
define(['../image/layer'], function(Layer) {
  // Segment viewer.
  function Viewer(imageURL, annotationURL, options) {
    if (typeof options === "undefined") options = {};
    this.colormap = options.colormap || [[255, 255, 255], [255, 0, 0]];
    this.labels = options.labels;
    this._createLayers(imageURL, annotationURL, options);
    var viewer = this;
    this.layers.image.load(imageURL, {
      width: options.width,
      height: options.height,
      onload: function () { viewer._initializeIfReady(options); }
    });
    this.layers.visualization.load(annotationURL, {
      width: options.width,
      height: options.height,
      imageSmoothingEnabled: false,
      onload: function () { viewer._initializeIfReady(options); },
      onerror: options.onerror
    });
    if (options.overlay)
      viewer.addOverlay(options.overlay);
  }

  Viewer.prototype._createLayers = function (imageURL,
                                             annotationURL,
                                             options) {
    var onload = options.onload;
    delete options.onload;
    this.container = document.createElement("div");
    this.container.classList.add("segment-viewer-container");
    this.layers = {
      image: new Layer(options),
      visualization: new Layer(options)
    };
    options.onload = onload;
    for (var key in this.layers) {
      var canvas = this.layers[key].canvas;
      canvas.classList.add("segment-viewer-layer");
      this.container.appendChild(canvas);
    }
    this._unloadedLayers = Object.keys(this.layers).length;
    this._resizeLayers(options);
  };

  Viewer.prototype._resizeLayers = function (options) {
    this.width = options.width || this.layers.image.canvas.width;
    this.height = options.height || this.layers.image.canvas.height;
    for (var key in this.layers) {
      if (key !== "image") {
        var canvas = this.layers[key].canvas;
        canvas.width = this.width;
        canvas.height = this.height;
      }
    }
    this.container.style.width = this.width + "px";
    this.container.style.height = this.height + "px";
  };

  Viewer.prototype._initializeIfReady = function (options) {
    if (--this._unloadedLayers > 0)
      return;
    this._resizeLayers(options);
    var viewer = this;
    this.layers.visualization.process(function () {
      var uniqueIndex = getUniqueIndex(this.imageData.data);
      this.applyColormap(viewer.colormap);
      this.setAlpha(192);
      this.render();
      if (viewer.labels)
        viewer.addLegend(uniqueIndex.filter(function (x) {
          return (options.excludedLegends || []).indexOf(x) < 0;
        }));
    });
  };

  Viewer.prototype.addOverlay = function (text) {
    var overlayContainer = document.createElement("div");
    overlayContainer.classList.add("segment-viewer-overlay-container");
    if (text)
      overlayContainer.appendChild(document.createTextNode(text));
    this.container.appendChild(overlayContainer);
  };

  Viewer.prototype.addLegend = function (index) {
    var legendContainer = document.createElement("div"),
        i;
    if (typeof index === "undefined") {
      index = [];
      for (i = 0; i < this.labels.length; ++i)
        index.push(i);
    }
    legendContainer.classList.add("segment-viewer-legend-container");
    for (i = 0; i < index.length; ++i) {
      var label = this.labels[index[i]],
          color = this.colormap[index[i]],
          legendItem = document.createElement("div"),
          colorbox = document.createElement("span"),
          legendLabel = document.createElement("span");
      colorbox.classList.add("segment-viewer-legend-colorbox");
      colorbox.style.backgroundColor = "rgb(" + color.join(",") + ")";
      legendItem.classList.add("segment-viewer-legend-item");
      legendLabel.appendChild(document.createTextNode(" " + label));
      legendLabel.classList.add("segment-viewer-legend-label");
      legendItem.appendChild(colorbox);
      legendItem.appendChild(legendLabel);
      legendContainer.appendChild(legendItem);
    }
    this.container.appendChild(legendContainer);
  };

  var getUniqueIndex = function (data) {
    var uniqueIndex = [];
    for (var i = 0; i < data.length; i += 4) {
      if (uniqueIndex.indexOf(data[i]) < 0) {
        uniqueIndex.push(data[i]);
      }
    }
    return uniqueIndex.sort(function (a, b) { return a - b; });
  };

  return Viewer;
});
