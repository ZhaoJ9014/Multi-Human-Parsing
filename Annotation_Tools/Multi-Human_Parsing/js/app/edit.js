/** Editor page renderer.
 */
define(['../image/layer',
        '../helper/segment-annotator',
        '../helper/util'],
function(Layer, Annotator, util) {
  // Create the navigation menu.
  function createNavigationMenu(params, data, annotator) {
    var navigationMenu = document.createElement("p"),
        navigation = createNavigation(params, data),
        idBlock = document.createElement("div");
    idBlock.className = "edit-top-menu-block";
    idBlock.appendChild(
        document.createTextNode(" ID = " + params.id));
    navigationMenu.appendChild(navigation);
    navigationMenu.appendChild(idBlock);
    return navigationMenu;
  }

  // Create the page navigation.
  function createNavigation(params, data) {
    var id = parseInt(params.id, 10),
        container = document.createElement("div"),
        indexAnchor = document.createElement("a"),
        indexAnchorText = document.createTextNode("Index"),
        prevAnchorText = document.createTextNode("Prev"),
        nextAnchorText = document.createTextNode("Next"),
        prevAnchor, nextAnchor;
    indexAnchor.href = util.makeQueryParams({ view: "index" });
    indexAnchor.appendChild(indexAnchorText);
    if (id > 0) {
      prevAnchor = document.createElement("a");
      prevAnchor.appendChild(prevAnchorText);
      prevAnchor.href = util.makeQueryParams(params, {
        id: id - 1
      });
    }
    else
      prevAnchor = prevAnchorText;
    if (id < data.imageURLs.length - 1) {
      nextAnchor = document.createElement("a");
      nextAnchor.appendChild(nextAnchorText);
      nextAnchor.href = util.makeQueryParams(params, {
        id: id + 1
      });
    }
    else
      nextAnchor = nextAnchorText;
    container.appendChild(prevAnchor);
    container.appendChild(document.createTextNode(" "));
    container.appendChild(indexAnchor);
    container.appendChild(document.createTextNode(" "));
    container.appendChild(nextAnchor);
    container.classList.add("edit-top-menu-block");
    return container;
  }

  // Create the main content block.
  function createMainDisplay(params, data, annotator, imageLayer) {
    var container = document.createElement("div"),
        imageContainerSpacer = document.createElement("div"),
        imageContainer = document.createElement("div"),
        annotatorTopMenu = createImageTopMenu(params, data, annotator),
        annotatorContainer = document.createElement("div"),
        sidebarSpacer = document.createElement("div"),
        sidebarContainer = document.createElement("div"),
        sidebar = createSidebar(params, data, annotator);
    imageContainerSpacer.className = "edit-image-top-menu";
    imageContainer.className = "edit-image-display";
    imageContainer.appendChild(imageContainerSpacer);
    imageContainer.appendChild(imageLayer.canvas);
    annotatorContainer.className = "edit-image-display";
    annotatorContainer.appendChild(annotatorTopMenu);
    annotatorContainer.appendChild(annotator.container);
    sidebarSpacer.className = "edit-image-top-menu";
    sidebarContainer.className = "edit-image-display";
    sidebarContainer.appendChild(sidebarSpacer);
    sidebarContainer.appendChild(sidebar);
    container.className = "edit-main-container";
    container.appendChild(imageContainer);
    container.appendChild(annotatorContainer);
    container.appendChild(sidebarContainer);
    return container;
  }

  // Create the menu above the editor.
  function createImageTopMenu(params, data, annotator) {
    var container = document.createElement("div"),
        zoomOutButton = document.createElement("div"),
        zoomInButton = document.createElement("div"),
        spacer1 = document.createElement("span"),
        finerButton = document.createElement("div"),
        boundaryButton = document.createElement("div"),
        coarserButton = document.createElement("div"),
        spacer2 = document.createElement("span"),
        alphaMinusButton = document.createElement("div"),
        imageButton = document.createElement("div"),
        alphaPlusButton = document.createElement("div");
    zoomOutButton.appendChild(document.createTextNode("-"));
    zoomOutButton.classList.add("edit-image-top-button");
    zoomOutButton.addEventListener("click", function () {
      annotator.zoomOut();
    });
    zoomInButton.appendChild(document.createTextNode("zoom +"));
    zoomInButton.classList.add("edit-image-top-button");
    zoomInButton.addEventListener("click", function () {
      annotator.zoomIn();
    });
    spacer1.className = "edit-image-top-spacer";
    boundaryButton.id = "boundary-button";
    boundaryButton.className = "edit-image-top-button";
    boundaryButton.appendChild(document.createTextNode("boundary"));
    boundaryButton.addEventListener("click", function () {
      if (boundaryFlashTimeoutID)
        window.clearTimeout(boundaryFlashTimeoutID);
      if (boundaryButton.classList.contains("edit-image-top-button-enabled"))
        annotator.hide("boundary");
      else
        annotator.show("boundary");
      boundaryButton.classList.toggle("edit-image-top-button-enabled");
    });
    finerButton.appendChild(document.createTextNode("-"));
    finerButton.className = "edit-image-top-button";
    finerButton.addEventListener("click", function () {
      annotator.finer();
      boundaryFlash();
    });
    coarserButton.appendChild(document.createTextNode("+"));
    coarserButton.className = "edit-image-top-button";
    coarserButton.addEventListener("click", function () {
      annotator.coarser();
      boundaryFlash();
    });
    spacer2.className = "edit-image-top-spacer";
    alphaMinusButton.className = "edit-image-top-button";
    alphaMinusButton.appendChild(document.createTextNode("-"));
    alphaMinusButton.addEventListener("click", function () {
      annotator.moreAlpha();
    });
    imageButton.className = "edit-image-top-button " +
                            "edit-image-top-button-enabled";
    imageButton.appendChild(document.createTextNode("image"));
    imageButton.addEventListener("click", function () {
      if (imageButton.classList.contains("edit-image-top-button-enabled"))
        annotator.hide("image");
      else
        annotator.show("image");
      imageButton.classList.toggle("edit-image-top-button-enabled");
    });
    alphaPlusButton.className = "edit-image-top-button";
    alphaPlusButton.appendChild(document.createTextNode("+"));
    alphaPlusButton.addEventListener("click", function () {
      annotator.lessAlpha();
    });
    //
    container.className = "edit-image-top-menu";
    container.appendChild(zoomOutButton);
    container.appendChild(zoomInButton);
    container.appendChild(spacer1);
    container.appendChild(finerButton);
    container.appendChild(boundaryButton);
    container.appendChild(coarserButton);
    container.appendChild(spacer2);
    container.appendChild(alphaMinusButton);
    container.appendChild(imageButton);
    container.appendChild(alphaPlusButton);
    return container;
  }

  // Set up the automatic flash of boundary.
  var boundaryFlashTimeoutID = null;
  function boundaryFlash() {
    var boundaryButton = document.getElementById("boundary-button");
    if (boundaryFlashTimeoutID) {
      window.clearTimeout(boundaryFlashTimeoutID);
      boundaryFlashTimeoutID = window.setTimeout(function() {
        boundaryButton.click();
        boundaryFlashTimeoutID = null;
      }, 1000);
    }
    else if (!boundaryButton.classList.contains(
             "edit-image-top-button-enabled")) {
      boundaryButton.click();
      boundaryFlashTimeoutID = window.setTimeout(function() {
        boundaryButton.click();
        boundaryFlashTimeoutID = null;
      }, 1000);
    }
  }

  // Create the sidebar.
  function createSidebar(params, data, annotator) {
    var container = document.createElement("div"),
        labelPicker = createLabelPicker(params, data, annotator),
        spacer1 = document.createElement("div"),
        undoButton = document.createElement("div"),
        redoButton = document.createElement("div"),
        spacer2 = document.createElement("div"),
        denoiseButton = document.createElement("div"),
        spacer3 = document.createElement("div"),
        superpixelToolButton = document.createElement("div"),
        spacer4 = document.createElement("div"),
        polygonToolButton = document.createElement("div"),
        spacer5 = document.createElement("div"),
        brushToolButton = document.createElement("div"),
        spacer6 = document.createElement("div"),
        manualParagraph = document.createElement("p"),
        spacer7 = document.createElement("div"),
        exportButton = document.createElement("input"),
        manualText;
    exportButton.type = "submit";
    exportButton.value = "export";
    exportButton.className = "edit-sidebar-submit";
    exportButton.addEventListener("click", function () {
      var filename = (data.annotationURLs) ?
          data.annotationURLs[params.id].split(/[\\/]/).pop() :
          params.id + ".png";
      downloadURI(annotator.export(), filename);
    });
    spacer1.className = "edit-sidebar-spacer";
    undoButton.className = "edit-sidebar-button";
    undoButton.appendChild(document.createTextNode("undo"));
    undoButton.addEventListener("click", function () { annotator.undo(); });
    redoButton.className = "edit-sidebar-button";
    redoButton.appendChild(document.createTextNode("redo"));
    redoButton.addEventListener("click", function () { annotator.redo(); });
    spacer2.className = "edit-sidebar-spacer";
    denoiseButton.className = "edit-sidebar-button";
    denoiseButton.appendChild(document.createTextNode("denoise"));
    denoiseButton.addEventListener("click", function () {
      annotator.denoise();
    });
    superpixelToolButton.className = "edit-sidebar-button";
    superpixelToolButton.appendChild(
      document.createTextNode("Superpixel tool"));
    superpixelToolButton.addEventListener("click", function () {
      polygonToolButton.classList.remove("edit-sidebar-button-selected");
      brushToolButton.classList.remove("edit-sidebar-button-selected");
      superpixelToolButton.classList.add("edit-sidebar-button-selected");
      annotator._setMode("superpixel");
    });
    superpixelToolButton.classList.add("edit-sidebar-button-selected");
    polygonToolButton.className = "edit-sidebar-button";
    polygonToolButton.appendChild(document.createTextNode("Polygon tool"));
    polygonToolButton.addEventListener("click", function () {
      superpixelToolButton.classList.remove("edit-sidebar-button-selected");
      brushToolButton.classList.remove("edit-sidebar-button-selected");
      polygonToolButton.classList.add("edit-sidebar-button-selected");
      annotator._setMode("polygon");
    });

    brushToolButton.classList.add("edit-sidebar-button-selected");
    brushToolButton.className = "edit-sidebar-button";
    brushToolButton.appendChild(document.createTextNode("Brush tool"));
    brushToolButton.addEventListener("click", function () {
      superpixelToolButton.classList.remove("edit-sidebar-button-selected");
      polygonToolButton.classList.remove("edit-sidebar-button-selected");
      brushToolButton.classList.add("edit-sidebar-button-selected");

      annotator._setMode("brush");
    });


    spacer3.className = "edit-sidebar-spacer";
    manualParagraph.appendChild(document.createTextNode("ctrl: toggle mode"));
    manualParagraph.appendChild(document.createElement("br"));
    manualParagraph.appendChild(document.createElement("br"));
    manualParagraph.appendChild(document.createTextNode("+Superpixel tool:"));
    manualParagraph.appendChild(document.createElement("br"));
    manualParagraph.appendChild(document.createTextNode("left: mark"));
    manualParagraph.appendChild(document.createElement("br"));
    manualParagraph.appendChild(document.createTextNode("right: pick label"));
    manualParagraph.appendChild(document.createElement("br"));
    manualParagraph.appendChild(document.createElement("br"));
    manualParagraph.appendChild(document.createTextNode("+Polygon tool:"));
    manualParagraph.appendChild(document.createElement("br"));
    manualParagraph.appendChild(document.createTextNode("left: draw line"));
    manualParagraph.appendChild(document.createElement("br"));
    manualParagraph.appendChild(document.createTextNode("right: abort"));
    spacer4.className = "edit-sidebar-spacer";
    container.className = "edit-sidebar";
    container.appendChild(labelPicker);
    container.appendChild(spacer1);
    container.appendChild(undoButton);
    container.appendChild(redoButton);
    container.appendChild(spacer2);
    container.appendChild(denoiseButton);
    container.appendChild(spacer3);
    container.appendChild(polygonToolButton);
    container.appendChild(superpixelToolButton);
    container.appendChild(brushToolButton);
    container.appendChild(manualParagraph);
    //container.appendChild(spacer4);
    container.appendChild(exportButton);
    return container;
  }

  function createLabelButton(data, value, index, annotator) {
    var colorBox = document.createElement("span"),
        labelText = document.createElement("span"),
        pickButton = document.createElement("div"),
        popupButton = document.createElement("div"),
        popupContainer = document.createElement("div");
    colorBox.className = "edit-sidebar-legend-colorbox";
    colorBox.style.backgroundColor =
        "rgb(" + data.colormap[index].join(",") + ")";
    labelText.appendChild(document.createTextNode(value));
    labelText.className = "edit-sidebar-legend-label";
    popupButton.appendChild(document.createTextNode("+"));
    popupButton.className = "edit-sidebar-popup-trigger";
    popupButton.addEventListener("click",  function () {
      popupContainer.classList.toggle("edit-sidebar-popup-active");
    });
    popupContainer.className = "edit-sidebar-popup";
    popupContainer.appendChild(
      createRelabelSelector(data, index, annotator, popupContainer)
      );
    popupContainer.addEventListener("click", function (event) {
      event.preventDefault();
    });
    pickButton.appendChild(colorBox);
    pickButton.appendChild(labelText);
    pickButton.appendChild(popupButton);
    pickButton.appendChild(popupContainer);
    pickButton.id = "label-" + index + "-button";
    pickButton.className = "edit-sidebar-button";
    pickButton.addEventListener("click", function () {
      var className = "edit-sidebar-button-selected";
      annotator.currentLabel = index;
      var selectedElements = document.getElementsByClassName(className);
      for (var i = 0; i < selectedElements.length; ++i)
        selectedElements[i].classList.remove(className);
      pickButton.classList.add(className);
    });
    pickButton.addEventListener('mouseenter', function () {
      if (!document.getElementsByClassName("edit-sidebar-popup-active").length)
        annotator.highlightLabel(index);
    });
    pickButton.addEventListener('mouseleave', function () {
      if (!document.getElementsByClassName("edit-sidebar-popup-active").length)
        annotator.unhighlightLabel();
    });
    return pickButton;
  }

    // Write the brush tool
  Annotator.prototype.brush = function (pos, label) {
    var offsets = [], labels = [];
    for (var y = -2; y <= 2; y++) {
      for (var x = -2; x <= 2; x++) {
        // it is circle bitches
        if ((x*x + y*y) > 7) continue;
        var offset = 4 * ((pos[1]+y) * this.layers.visualization.canvas.width + (pos[0]+x));
        offsets.push(offset);
        labels.push(label);
      }
    }
    this._updateAnnotation(offsets, labels);
    this.layers.visualization.render();
    if (typeof this.onchange === "function")
      this.onchange.call(this);
  };

  // Hightlight legend labels.
  function highlightLabel(label) {
    var highlightClass = "edit-sidebar-button-highlight",
        elements = document.getElementsByClassName(highlightClass);
    for (var i = 0; i < elements.length; ++i)
      elements[i].classList.remove(highlightClass);
    var pickButton = document.getElementById("label-" + label + "-button");
    if (pickButton)
      pickButton.classList.add(highlightClass);
  }

  // Create the label picker button.
  function createLabelPicker(params, data, annotator) {
    var container = document.createElement("div");
    container.className = "edit-sidebar-label-picker";
    for (var i = 0; i < data.labels.length; ++i) {
      var labelButton = createLabelButton(data, data.labels[i], i, annotator);
      if (i === 0) {
        annotator.currentLabel = 0;
        labelButton.classList.add("edit-sidebar-button-selected");
      }
      container.appendChild(labelButton);
    }
    window.addEventListener("click", cancelPopup, true);
    return container;
  }

  // Cancel popup.
  function cancelPopup(event) {
    var isOutsidePopup = true,
        target = event.target;
    while (target.parentNode) {
      isOutsidePopup = isOutsidePopup &&
                       !target.classList.contains("edit-sidebar-popup");
      target = target.parentNode;
    }
    if (isOutsidePopup) {
      var popups = document.getElementsByClassName(
          "edit-sidebar-popup-active");
      if (popups.length)
        for (var i = 0; i < popups.length; ++i)
          popups[i].classList.remove("edit-sidebar-popup-active");
    }
  }

  // Create the relabel selector.
  function createRelabelSelector(data, index, annotator, popupContainer) {
    var select = document.createElement("select"),
        firstOption = document.createElement("option");
    firstOption.appendChild(document.createTextNode("Change to"));
    select.appendChild(firstOption);
    for (var i = 0; i < data.labels.length; ++i) {
      if (i !== index) {
        var option = document.createElement("option");
        option.value = i;
        option.appendChild(document.createTextNode(data.labels[i]));
        select.appendChild(option);
      }
    }
    select.addEventListener("change", function (event) {
      var sourceLabel = index;
      var targetLabel = parseInt(event.target.value, 10);
      if (sourceLabel !== targetLabel) {
        var currentLabel = annotator.currentLabel;
        annotator.currentLabel = targetLabel;
        annotator.fill(sourceLabel);
        annotator.currentLabel = currentLabel;
      }
      popupContainer.classList.remove("edit-sidebar-popup-active");
      firstOption.selected = true;
      event.preventDefault();
    });
    return select;
  }

  // Download trick.
  function downloadURI(uri, filename) {
    var anchor = document.createElement("a");
    anchor.style.display = "none";
    anchor.target = "_blank"; // Safari doesn't work.
    anchor.download = filename;
    anchor.href = uri;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
  }

  // Entry point.
  function render(data, params) {
    var id = parseInt(params.id, 10);
    if (isNaN(id))
      throw("Invalid id");
    var annotator = new Annotator(data.imageURLs[id], {
          width: params.width,
          height: params.height,
          colormap: data.colormap,
          superpixelOptions: { method: "slic", regionSize: 25 },
          onload: function () {
            if (data.annotationURLs)
              annotator.import(data.annotationURLs[id]);
            annotator.hide("boundary");
            boundaryFlash();
          },
          onchange: function () {
            var activeLabels = this.getUniqueLabels(),
                legendClass = "edit-sidebar-legend-label",
                legendActiveClass = "edit-sidebar-legend-label-active",
                elements = document.getElementsByClassName(legendClass),
                i;
            for (i = 0; i < elements.length; ++i)
              elements[i].classList.remove(legendActiveClass);
            for (i = 0; i < activeLabels.length; ++i)
              elements[activeLabels[i]].classList.add(legendActiveClass);
          },
          onrightclick: function (label) {
            document.getElementById("label-" + label + "-button").click();
          },
          onmousemove: highlightLabel
        }),
        imageLayer = new Layer(data.imageURLs[id], {
          width: params.width,
          height: params.height
        });
    document.body.appendChild(createNavigationMenu(params, data, annotator));
    document.body.appendChild(createMainDisplay(params,
                                                data,
                                                annotator,
                                                imageLayer));
  }

  return render;
});
