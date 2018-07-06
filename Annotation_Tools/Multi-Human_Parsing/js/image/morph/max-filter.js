/** Max filter for an index image.
 *
 * Copyright 2015  Kota Yamaguchi
 */
define(["./neighbor-map"], function (NeighborMap) {
  function findDominantLabel(data, neighbors) {
    var histogram = {},
        i, label;
    for (i = 0; i < neighbors.length; ++i) {
      label = data[neighbors[i]];
      if (histogram[label])
        ++histogram[label];
      else
        histogram[label] = 1;
    }
    var labels = Object.keys(histogram),
        count = 0,
        dominantLabel = null;
    for (i = 0; i < labels.length; ++i) {
      label = labels[i];
      if (histogram[label] > count) {
        dominantLabel = parseInt(label, 10);
        count = histogram[label];
      }
    }
    return dominantLabel;
  }

  function maxFilter(indexImage, options) {
    options = options || {};
    var neighbors = options.neighbors || [[-1, -1], [-1, 0], [-1, 1],
                                          [ 0, -1], [ 0, 0], [ 0, 1],
                                          [ 1, -1], [ 1, 0], [ 1, 1]],
        result = new Int32Array(indexImage.data.length),
        neighborMap = new NeighborMap(indexImage.width,
                                      indexImage.height,
                                      neighbors);
    for (var i = 0; i < indexImage.data.length; ++i)
      result[i] = findDominantLabel(indexImage.data,
                                    neighborMap.get(i));
    return {
      width: indexImage.width,
      height: indexImage.height,
      data: result
    };
  }

  return maxFilter;
});
