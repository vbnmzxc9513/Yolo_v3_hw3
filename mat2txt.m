load train/digitStruct.mat
[col, row] = size(digitStruct);
output = '';
for i = 1:row
    output = [output './train/' digitStruct(i).name ' '];
    im = imread([strcat('./train/',digitStruct(i).name)]);
    [x, num] = size(digitStruct(i).bbox);
    for j = 1:num
        [height, width] = size(im);
        aa = max(digitStruct(i).bbox(j).top+1,1);
        bb = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height-1);
        cc = max(digitStruct(i).bbox(j).left+1,1);
        dd = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width-1);   
        output = [output int2str(cc) ',' int2str(aa) ',' int2str(dd) ',' int2str(bb) ',' int2str(digitStruct(i).bbox(j).label)];
        if j == num
            output = [output '\n'];
        else
            output = [output ' '];
        end
    end
end

fid = fopen('output.txt', 'wt');
fprintf(fid, output);
fclose(fid);
