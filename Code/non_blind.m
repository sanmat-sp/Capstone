rgbImage= imread("C:\Users\sanma\OneDrive\Documents\MATLAB\Real_time_image.jpg");
rgbImage1=rgbImage;
fgrprt= imread("C:\Users\sanma\OneDrive\Documents\MATLAB\fingerprint3.tif");
BW_fgrprt1=rgb2gray(fgrprt);
BW_fgrprt=imbinarize(BW_fgrprt1);

datasource = "caps";
username = "root";
password = "2802";
conn = database(datasource,username,password);

prompt="What is the document number? ";
doc_no=input(prompt);
fgrprt_1 = reshape(BW_fgrprt.',1,[]);
fgrprt_2 = uint8(fgrprt_1);

greenChannel=rgbImage(:,:,2);
[LL, LH, HL, HH]=dwt2(greenChannel, "haar");
I=dct2(LL);
original_image=I;
original_image1 = reshape(original_image.',1,[]);
original_image1 = uint8(original_image1);
blockSizeR =128;
blockSizeC =128;
[rows, columns, numberOfColorBands] = size(I);

wholeBlockRows = floor(rows / blockSizeR);
blockVectorR = [blockSizeR * ones(1, wholeBlockRows), rem(rows, blockSizeR)];

wholeBlockCols = floor(columns / blockSizeC);
blockVectorC = [blockSizeC * ones(1, wholeBlockCols), rem(columns, blockSizeC)];

blocks = mat2cell(I, blockVectorR, blockVectorC);

WR = []; mrow = []; A=[WR,mrow];i=1;j=1;
for indx = 1: 2
    for indy = 1: 2
                   A(j,i)= test_entropy(blocks{j,i});
              j=j+1;
    end
    i=i+1;  j=1;
end 

B=sort(A,"descend");
res=find(A==B(1));
result=res(1);
bst_blk=blocks{result};

data=table(doc_no,fgrprt_2,original_image1,result,'VariableNames',{'img_id' 'img' 'll' 'block_no'});
sqlwrite(conn,"capstone.non_blind",data)

[Uw, Sw, Vw]=svd(double(BW_fgrprt));
[Uh, Sh, Vh]=svd(bst_blk);
Sh_diag = diag(Sh);
Sw_diag = diag(Sw);

if (length(BW_fgrprt) >= length(bst_blk))
    Sh_diag(1:length(Sh), :) = Sh_diag(1:length(Sh), :) + Sw_diag(1:length(Sh), :);
elseif(length(BW_fgrprt) < length(bst_blk))
    Sh_diag(1:length(BW_fgrprt), :) = Sh_diag(1:length(BW_fgrprt), :) + Sw_diag(1:length(BW_fgrprt), :);
end

Sh(logical(eye(size(Sh)))) = Sh_diag;
bstblk_wtrmkd=Uh * Sh * Vh';
blocks{result}=bstblk_wtrmkd;
I=cell2mat(blocks);
greenChannel1=idct2(I);
greenChannel1=idwt2(greenChannel1,LH,HL,HH,"haar");
rgbImage(:,:,2)=greenChannel1;

attacks = {'No Attack', 'Gaussian low-pass filter', 'Median', 'Gaussian noise',...
    'Salt and pepper noise','Speckle noise', ...
    'Sharpening attack', 'Histogram equalization',...
    'Average filter', 'Motion blur','Scaling','Rotate','Morph'};
params = [0; 3; 3; 0.001; 0; 0; 0.8; 0; 0; 0; 1.5; 90; 0];

PSNR_CI=[];
SSIM_CI=[];
NCC_CI=[];
BER_CI=[];
PSNR_EI=[];
SSIM_EI=[];
NCC_EI=[];
BER_EI=[];
Au=[];
mse_array=[];

%PSO-------------------------------------------------------------------------------------------------------------
for j=1:length(attacks)
    attack = string(attacks(j));
    param = params(j);
    watermarked_image = Attacks(rgbImage,attack,param);
    
    numParticles = 30;
    numIterations = 50;
    minParam=0.0001;
    maxParam=0.001; 
    paramBounds=[0.001,0.0001];
    swarmSize = numParticles;
    particles = rand(numParticles, 1) * (paramBounds(2) - paramBounds(1)) + paramBounds(1);
    velocity = rand(numParticles, 1) * 0.2; 
    
    bestPositions = particles;
    bestFitness = inf(size(particles));
    globalBestPosition = [];
    globalBestFitness = inf;
    
    for iteration = 1:numIterations
        for particleIdx = 1:numParticles
            fitness = evaluateFitness(watermarked_image, particles(particleIdx));
            
            if fitness < bestFitness(particleIdx)
                bestFitness(particleIdx) = fitness;
                bestPositions(particleIdx) = particles(particleIdx);
            end
            
            if fitness < globalBestFitness
                globalBestFitness = fitness;
                globalBestPosition = particles(particleIdx);
            end
        end
        
        inertiaWeight = 0.7;
        cognitiveWeight = 1.5;
        socialWeight = 1.5;
        velocity = zeros(numParticles, 1);
        for particleIndex = 1:numParticles
            velocity(particleIndex) = inertiaWeight * velocity(particleIndex) + cognitiveWeight * rand() * (bestPositions(particleIndex) - particles(particleIndex)) + socialWeight * rand() * (globalBestPosition - particles(particleIndex));
            
            particles(particleIndex) = particles(particleIndex) + velocity(particleIndex);
            
            particles(particleIndex) = max(minParam, min(maxParam, particles(particleIndex)));
        end
    end
    
    bestParameter = globalBestPosition;
    enhanced_watermarked_image = GlowpassFilter(watermarked_image, bestParameter);

    mse = mean(mean((im2double(rgbImage1) - im2double(enhanced_watermarked_image)).^2, 1), 2);
    PSNR = 10 * log10(1 ./ mean(mse,3));
    SSIM = ssim(double(rgbImage1),double(enhanced_watermarked_image));
    NC=normalizedCorrelation(rgbImage1,enhanced_watermarked_image);
    BER=bitErrorRate(rgbImage1,enhanced_watermarked_image);
    
    PSNR_CI=[PSNR_CI,PSNR];
    SSIM_CI=[SSIM_CI,SSIM];
    NCC_CI=[NCC_CI,NC];
    BER_CI=[BER_CI,BER];
    
    figure(1);
    subplot(4,4,j);
    imshow(enhanced_watermarked_image);
    title(attack)
    xlabel('PSNR: '+string(PSNR));
    sgtitle('Watermarked Cover Image (Non-blind method)')

    I=enhanced_watermarked_image;
    greenChannel2=I(:,:,2);
    [LL, LH, HL, HH]=dwt2(greenChannel2,"haar");
    greenChannel2=dct2(LL);
    greenChannel2= greenChannel2-original_image;
    blockSizeR = 128;
    blockSizeC = 128;
    [rows, columns, numberOfColorBands] = size(greenChannel2);

    wholeBlockRows = floor(rows / blockSizeR);
    blockVectorR = [blockSizeR * ones(1, wholeBlockRows), rem(rows, blockSizeR)];

    wholeBlockCols = floor(columns / blockSizeC);
    blockVectorC = [blockSizeC * ones(1, wholeBlockCols), rem(columns, blockSizeC)];

    blocks1 = mat2cell(greenChannel2, blockVectorR, blockVectorC);
    blocks2 = mat2cell(LH, blockVectorR, blockVectorC);
    blocks3 = mat2cell(HL, blockVectorR, blockVectorC);
    blocks4 = mat2cell(HH, blockVectorR, blockVectorC);

    bst_blk1=blocks1{result};
    bst_blk2=blocks2{result};
    bst_blk3=blocks3{result};
    bst_blk4=blocks4{result};

    greenchannel_extract=bst_blk1;
    I1=idct2(greenchannel_extract);
    I1=idwt2(I1,bst_blk2,bst_blk3,bst_blk4,"haar");
    [Ucw, Scw, Vcw] = svd(double(I1));
    [Uw_x, Sw_x, Vw_x] = svd(double(BW_fgrprt));
    BC_singularValues = zeros(length(double(BW_fgrprt)));
    Shh_diag = diag(BC_singularValues);
    Scw_diag = diag(Scw);
 
    if (length(BW_fgrprt) >= length(bst_blk1))
        Shh_diag(1:length(Scw), :) = Scw_diag;
    elseif (length(BW_fgrprt) <length(bst_blk1))
        Shh_diag(1:length(BW_fgrprt), :) = Scw_diag(1:length(BW_fgrprt), :);
    end
        
    BC_singularValues(logical(eye(size(BC_singularValues)))) = Shh_diag;
    watermark_logo_extracted = Uw_x * BC_singularValues * Vw_x';
    watermark_logo_extracted=imbinarize(watermark_logo_extracted);

    mse = mean(mean((im2double(BW_fgrprt) - im2double(watermark_logo_extracted)).^2, 1), 2);
    PSNR = 10 * log10(1 ./ mean(mse,3));
    SSIM = ssim(double(BW_fgrprt),double(watermark_logo_extracted));
    NC=normalizedCorrelation(BW_fgrprt,watermark_logo_extracted);
    BER=bitErrorRate(BW_fgrprt,watermark_logo_extracted);
    
    PSNR_EI=[PSNR_EI,PSNR];
    SSIM_EI=[SSIM_EI,SSIM];
    NCC_EI=[NCC_EI,NC];
    BER_EI=[BER_EI,BER];
    mse_array=[mse_array,mse];

    figure(2);
    subplot(4,4,j);
    imshow(watermark_logo_extracted);
    title(attack)
    xlabel('PSNR: '+string(PSNR));
    sgtitle('Extracted Watermark Image (Non-blind method)')

    if mse<=0.005
        Au=[Au,"Images match"];
    else
        Au=[Au,"Images donot match"];
    end
end

%OUTPUT--------------------------------------------------------------------------------------------------------------------------------------------------------------
columnNames1={'S_No','Attack','PSNR_CI','SSIM_CI','NCC_CI','BER_CI'};
T1=table([1,2,3,4,5,6,7,8,9,10,11,12,13]', attacks', PSNR_CI', SSIM_CI', NCC_CI', BER_CI', 'VariableNames', columnNames1);
disp("PSNR, SSIM, NCC and BER of Watermarked Cover Image");
disp(T1);

columnNames2={'S_No','Attack','PSNR_EI','SSIM_EI','NCC_EI','BER_EI'};
T2=table([1,2,3,4,5,6,7,8,9,10,11,12,13]', attacks', PSNR_EI', SSIM_EI', NCC_EI', BER_EI', 'VariableNames', columnNames2);
disp("PSNR, SSIM, NCC and BER of Extracted Watermark Image");
disp(T2);

columnNames3={'S_No','Attack','Authentication','PSNR_EI','MSE_EI'};
T3=table([1,2,3,4,5,6,7,8,9,10,11,12,13]',attacks',Au',PSNR_EI',mse_array','VariableNames',columnNames3);
disp("PSNR, MSE and Authentication of Extracted Watermark Image");
disp(T3);

figure(3);
bar(1:length(attacks), [PSNR_CI;PSNR_EI]);
title('PSNR-CI and PSNR-EI against Attacks - Non-Blind Method');
xlabel('Attacks');
ylabel('PSNR (dB)');
legend('PSNR_{CI}','PSNR_{EI}');
set(gca, 'XTickLabel', attacks);
xtickangle(45);

% functions ---------------------------------------------------------------------------------------------------------------------------------------
function Entropy= test_entropy(x)
    [Height,Width]=size(x);
    [m,Binsx]=imhist(x);
    m=m/(Height*Width);
    H=abs(-sum(m.*log(m+1e-10))); % Visual entropy
    H2=sum(m.*exp(1-m)); 
    Entropy= H + H2 ;
    Entropy=round(Entropy,4); 
end

function histImageAttacked = histAttack(watermarked_image)
    histImageAttacked = histeq(watermarked_image);
end

function medianImageAttacked = medianAttack(watermarked_image,m)
    medianImageAttacked = medfilt3(watermarked_image,[m m m]);
end

function motionImageAttacked = motionAttack(watermarked_image)
    h = fspecial('motion',7,4);
    motionImageAttacked = imfilter(watermarked_image,h,'replicate');
end

function GaussNoiseImageAttacked = noiseGauss(watermarked_image,var)
    GaussNoiseImageAttacked = imnoise(watermarked_image, 'gaussian', 0,var);
end

function SaltPepperNoiseImageAttacked = noiseSaltPepper(watermarked_image)
    SaltPepperNoiseImageAttacked = imnoise(watermarked_image,'salt & pepper',0.001);
end

function SpeckleNoiseImageAttacked = noiseSpeckle(watermarked_image)
    SpeckleNoiseImageAttacked = imnoise(watermarked_image, 'speckle', 0.001);
end

function sharpenImageAttacked = sharpenAttack(watermarked_image,strength)
    sharpenImageAttacked = imsharpen(watermarked_image,'Amount',strength);
end

function averageImageAttacked = averageFilter(watermarked_image)
    h = fspecial('average',[3 3]);
    averageImageAttacked = imfilter(watermarked_image,h,'replicate');
end

function GlowPassFilterImageAttacked = GlowpassFilter(watermarked_image,sigma)
    h = fspecial('gaussian',[3 3],sigma);
    GlowPassFilterImageAttacked = imfilter(watermarked_image,h,'replicate');
end

function fitness = evaluateFitness(image, parameter)
    enhanced_image = GlowpassFilter(image, parameter);
    mse = mean(mean((im2double(image) - im2double(enhanced_image)).^2, 1), 2);
    PSNR = 10 * log10(1 ./ mean(mse, 3));
    fitness = -PSNR;
end

function nc=normalizedCorrelation(originalImage, watermarkedImage)
    if size(originalImage, 3) == 3
        o_image = rgb2gray(originalImage);
    else
        o_image=originalImage;
    end
    if size(watermarkedImage, 3) == 3
        w_image = rgb2gray(watermarkedImage);
    else
        w_image=watermarkedImage;
    end
    normalized_correlation = normxcorr2(o_image, w_image);
    [ypeak, xpeak] = find(normalized_correlation == max(normalized_correlation(:)));
    nc = normalized_correlation(ypeak, xpeak);
end

function BER=bitErrorRate(originalImage, watermarkedImage)
    if size(originalImage, 3) == 3
        o_image = rgb2gray(originalImage);
    else
        o_image=originalImage;
    end
    if size(watermarkedImage, 3) == 3
        w_image = rgb2gray(watermarkedImage);
    else
        w_image=watermarkedImage;
    end
    originalImageBinary = imbinarize(double(o_image));
    watermarkedImageBinary = imbinarize(double(w_image));
    numPixels = numel(originalImageBinary);
    bitErrors = sum(originalImageBinary(:) ~= watermarkedImageBinary(:));
    BER = bitErrors / numPixels;
end

function morphingWatermarkedImage = morphing(watermarkedImage)
    morphing_image = imread("C:\Users\sanma\OneDrive\Documents\MATLAB\lena_color_512.tif");
    [Rw, Cw, Cow]=size(watermarkedImage);
    [Rm, Cm, Com]=size(morphing_image);
    if (Rw~=Rm) || (Cw~=Cm) || (Cow~=Com)
        cropSize=[0,0,Cw,Rw];
        morphing_image=imcrop(morphing_image,cropSize);
    end
    numFrames = 30;
    morphedFrames = cell(1, numFrames);
    for frameIndex = 1:numFrames
        t = (frameIndex - 1) / (numFrames - 1);
        morphedFrame = uint8((1 - t) * double(watermarkedImage) + t * double(morphing_image));
        morphedFrames{frameIndex} = morphedFrame;
    end
    morphingWatermarkedImage = morphedFrames{numFrames};
end

function RotatedImage = RotateImage(image, angle_degrees)
    angle_radians = deg2rad(angle_degrees);
    [height, width, ~] = size(image);
    rotation_matrix = [cos(angle_radians), -sin(angle_radians); 
                       sin(angle_radians), cos(angle_radians)];
    new_corners = rotation_matrix * [1 1; width 1; width height; 1 height]';
    new_width = round(max(new_corners(1,:)) - min(new_corners(1,:)));
    new_height = round(max(new_corners(2,:)) - min(new_corners(2,:)));
    translation = [width; height] - [new_width; new_height];
    RotatedImage = imrotate(image, angle_degrees, 'bilinear', 'crop');
    crop_x = max(0, round(translation(1)));
    crop_y = max(0, round(translation(2)));
    if crop_x==0 && crop_y==0 
        RotatedImage = RotatedImage(crop_y + 1 : crop_y + height, crop_x + 1 : crop_x + width, :);
    elseif crop_x==1 && crop_y==1
        RotatedImage = RotatedImage(crop_y  : crop_y + height-1, crop_x : crop_x + width-1, :);
    end
end

function ScaledImage = ScaleImage(image, scale_factor)
    [height, width, ~] = size(image);
    new_height = round(height * scale_factor);
    new_width = round(width * scale_factor);
    ScaledImage = imresize(image, [new_height, new_width]);
    ScaledImage = imresize(ScaledImage, [height, width]);
end

function [watermarked_image] = Attacks(watermarked_image,attack,param)
switch attack
    case 'No Attack'
    case 'Median'
        watermarked_image = medianAttack(watermarked_image,param);
    case 'Gaussian noise'
        watermarked_image = noiseGauss(watermarked_image,param);
    case 'Salt and pepper noise'
        watermarked_image = noiseSaltPepper(watermarked_image);
    case 'Speckle noise'
        watermarked_image = noiseSpeckle(watermarked_image);
    case 'Sharpening attack'
        watermarked_image = sharpenAttack(watermarked_image,param);
    case 'Motion blur'
        watermarked_image = motionAttack(watermarked_image);
    case 'Average filter'
        watermarked_image = averageFilter(watermarked_image);
    case 'Gaussian low-pass filter'
        watermarked_image = GlowpassFilter(watermarked_image,param);
    case 'Histogram equalization'
        watermarked_image = histAttack(watermarked_image);
    case 'Morph'
        watermarked_image = morphing(watermarked_image);
    case 'Rotate'
        watermarked_image=RotateImage(watermarked_image, param);
    case 'Scaling'
        watermarked_image=ScaleImage(watermarked_image, param);
    otherwise
        errordlg('Please specify attack!');
end
end