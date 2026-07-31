
module fin(height, chord, thickness, type) {
  if (type == "trapezoidal") {
    linear_extrude(thickness) polygon([[0,0], [chord,0], [chord/2,height], [0,height]]);
  } else if (type == "elliptical") {
    linear_extrude(thickness) scale([chord/2, height/2]) circle(1);
  }
}
fin($height, $chord, $thickness, $type);
