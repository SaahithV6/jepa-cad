
module nose_cone(diameter, length, type) {
  if (type == "ogive") {
    cylinder(d=diameter, h=length);
  } else if (type == "conical") {
    cylinder(d1=diameter, d2=0, h=length);
  } else if (type == "parabolic") {
    for (i = [0:0.1:length]) {
      translate([0, 0, i])
        cylinder(d=diameter*(1-(i/length)^2), h=0.1);
    }
  }
}
nose_cone($diameter, $length, $type);
