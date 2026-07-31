
module tube(diameter, length, thickness) {
  difference() {
    cylinder(d=diameter, h=length);
    cylinder(d=diameter-2*thickness, h=length);
  }
}
tube($diameter, $length, $thickness);
