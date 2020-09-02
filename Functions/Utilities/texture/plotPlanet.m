function [CelBody] = plotPlanet(coord,ibody)

switch ibody
    case 1 % Mercury
        [R,~,~,~,~] = bulkplanet(1);
        map = imread('Mercury.jpg','jpg');
    case 2 % Venus
        [R,~,~,~,~] = bulkplanet(2);
        map = imread('Venus.jpg','jpg');
    case 3 % Earth
        [R,~,~,~,~] = bulkplanet(3);
        map = imread('Earth.jpg','jpg');
    case 4 % Mars
        [R,~,~,~,~] = bulkplanet(4);
        map = imread('Mars.jpg','jpg');
    case 5 % Jupiter
        [R,~,~,~,~] = bulkplanet(5);
        map = imread('Jupiter.jpg','jpg');
    case 6 % Saturn
        [R,~,~,~,~] = bulkplanet(6);
        map = imread('Saturn.jpg','jpg');
    case 7 % Uranus
        [R,~,~,~,~] = bulkplanet(7);
        map = imread('Uranus.jpg','jpg');
    case 8 % Neptune
        [R,~,~,~,~] = bulkplanet(8);
        map = imread('Naptune.jpg','jpg');
    otherwise
end

R_phi_equa = R;
R_phi_polar = R;
props.FaceColor = 'texture';
props.EdgeColor = 'none';
props.FaceLighting = 'phong';
props.Cdata = map;
[XX, YY, ZZ] = ellipsoid(-coord(1), -coord(2), -coord(3), ...
    R_phi_equa, R_phi_equa, R_phi_polar, 30);
[CelBody] = surface(-XX, -YY, -ZZ, props);

end