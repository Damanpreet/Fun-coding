image = imresize(imread('Lenna.png'), [224, 224]);
img = double(rgb2gray(image));
%size(img)

% DCT
T = dctmtx(8);
dct = @(block_struct) T * block_struct.data * T';
dct_res = blockproc(img, [8,8], dct);
dct_res = floor(dct_res);

% Quantization matrix
quant_mat = [16 11 10 16 24 40 51 61;
            12 12 14 19 26 58 60 55;
            14 13 16 24 40 57 69 56; 
            14 17 22 29 51 87 80 62;
            18 22 37 56 68 109 103 77;
            24 35 55 64 81 104 113 92;
            49 64 78 87 103 121 120 101;
            72 92 95 98 112 100 103 99];
        
% Quantize
quant = @(block_struct) block_struct.data ./ quant_mat;
% quant = @(block_struct) block_struct.data ./ N;
Quant_res = blockproc(dct_res, [8, 8], quant);
Quant_res = round(Quant_res);  %round the result
%imshow(Quant_res)

% Decoder
% Inverse Quantize
invert_quant = @(block_struct) block_struct.data .* quant_mat;
% invert_quant = @(block_struct) block_struct.data .* N;
Inv_quant_res = blockproc(Quant_res, [8, 8], invert_quant);

% Inverse DCT
idct = @(block_struct) T' * block_struct.data * T;
indct_res = blockproc(Inv_quant_res, [8, 8], idct);
indct_res = floor(indct_res);
% indct_res(1:5, 1:5)
% img(1:5, 1:5)

% -------------------------------------------------------------------
% Compression on RGB Image
imgycbcr = imresize(rgb2ycbcr(imread('Lenna.png')), [224, 224]);
size(imgycbcr)

N = 10;


lb = {'Y', 'Cb', 'Cr'};
for channel = 1:3
    subplot(1,3,channel)
    im = imgycbcr;
    im(:,:,setdiff(1:3,channel)) = intmax(class(im))/2;
    imshow(ycbcr2rgb(im))
    title([lb{channel} ' component'],'fontsize',16)
end

% downsample the illuminance
subplot(1, 2, 1)
imshow(image)
title('Original')
subplot(1, 2, 2)
Y = imgycbcr;
Y(:,:,1) = N*round(Y(:,:,1)/N);
imshow(ycbcr2rgb(Y))
title('Downsample illuminance');


% downsample only chrominance
subplot(1, 2, 1)
imshow(image)
title('Original')
subplot(1, 2, 2)
Y = imgycbcr;
Y(:,:,2) = N * round(Y(:,:,2)/N);
Y(:,:,3) = N * round(Y(:,:,3)/N);
imshow(ycbcr2rgb(Y))
title('Compressed')

A = zeros(size(Y));
B = A;

T = dctmtx(8);
for i=1:3
    img = double(Y(:,:,i)); % used the downsampled chrominance image
    
    % DCT
    dct = @(block_struct) T * block_struct.data * T';
    dct_res = blockproc(img, [8,8], dct);
    dct_res = floor(dct_res);

    % Quantize
    quant = @(block_struct) block_struct.data ./ quant_mat;
    Quant_res = blockproc(dct_res, [8, 8], quant);
    Quant_res = round(Quant_res);  %round the result
    
    % Inverse Quantize
    invert_quant = @(block_struct) block_struct.data .* quant_mat;
    Inv_quant_res = blockproc(Quant_res, [8, 8], invert_quant);

    % Inverse DCT
    idct = @(block_struct) T' * block_struct.data * T;
    indct_res = blockproc(Inv_quant_res, [8, 8], idct);
    indct_res = floor(indct_res);
    
    A(:,:,i) = Inv_quant_res;
    B(:,:,i) = indct_res;
end

subplot(1, 2, 1)
imshow(image)
title('Original')
subplot(1, 2, 2)
imshow(ycbcr2rgb(uint8(B)))
title('JPEG Image Compressed')

% Compressing the two luminance channels.
A = zeros(size(Y));
B = A;
N = 40;
T = dctmtx(8);
for i=1:3
    img = double(Y(:,:,i)); % used the downsampled chrominance image
    
    % DCT
    dct = @(block_struct) T * block_struct.data * T';
    dct_res = blockproc(img, [8,8], dct);
    dct_res = floor(dct_res);

    % Quantize
    if (i==1)
        quant = @(block_struct) block_struct.data ./ quant_mat;
    else
        quant = @(block_struct) block_struct.data ./ N;
    end
    
    Quant_res = blockproc(dct_res, [8, 8], quant);
    Quant_res = round(Quant_res);  %round the result
    
    % Inverse Quantize
    if (i==1)
        invert_quant = @(block_struct) block_struct.data .* quant_mat;
    else
        invert_quant = @(block_struct) block_struct.data .* N;
    end
    Inv_quant_res = blockproc(Quant_res, [8, 8], invert_quant);

    % Inverse DCT    
    idct = @(block_struct) T' * block_struct.data * T;
    indct_res = blockproc(Inv_quant_res, [8, 8], idct);
    indct_res = floor(indct_res);
    
    A(:,:,i) = Inv_quant_res;
    B(:,:,i) = indct_res;
end

subplot(1, 2, 1)
imshow(image)
title('Original')
subplot(1, 2, 2)
imshow(ycbcr2rgb(uint8(B)))
title('JPEG Image Compressed')
