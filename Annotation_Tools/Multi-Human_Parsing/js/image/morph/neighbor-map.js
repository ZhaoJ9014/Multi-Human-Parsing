/** Create a map of neighbor offsets.
 *
 *  var neighborMap = new NeighborMap(width, height);
 *  for (var i = 0; i < data.length; ++i) {
 *    var neighbors = neighborMap.get(i);
 *    for (var j = 0; j < neighbors.length; ++j) {
 *      var pixel = data[neighbors[j]];
 *    }
 *  }
 *
 * Copyright 2015  Kota Yamaguchi
 */
define(function () {
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

  return NeighborMap;
});
