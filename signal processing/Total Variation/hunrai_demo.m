%HUNRAI-DEBLURRING DEMO

%Load the "true" image. This has already been converted into double
%arithmnetic format from its original image format using im2double(DSC_XYZ.jpg);
load hunrai_example 

%You can display the whole image as
figure; image(true); daspect([1 1 1]);

%This is a 3-dimensional tensor of pixel intensities. It is however too big 
%to process so I will cut out a small section. Note that to keep things 
%simple I am assuming all images to be square, i.e. pixels in a column = pixels in a row. Say 
trues = true(3751:4250,4751:5250,:);
figure; image(trues); daspect([1 1 1]);

%For computating/processing each of the three image channels (tensor dimensions) 
%the image will be will be converted to a long vector by concatenating its columns

for i=1:3 
    x(:,i) = reshape(trues(:,:,i),size(trues,1)*size(trues,2),1); 
end

%Now that we have "reshaped" our clean image, we can blur it using a
%blurring kernel A. The first argument in blur is the number of pixels in a
%row of the image. The second argument is "radius" of the blur in pixels, and 
%the third argument is the weight of the blurr. The third argumemnt must be 
% smaller than the second. 
[A,~,~] = blur(size(trues,1),20,10);

%To see this blurring kernel "at a given pixel" of the image, plot the
%respective row of A using
figure; imagesc(reshape(A(135670,:),500,500)); daspect([1 1 1]);
%for the interior pixel number 135670 or
figure; imagesc(reshape(A(10,:),500,500)); daspect([1 1 1]);
%for the tenth pixel that happens to be on the boundary.

%The blur function has two other putputs b and x. If you don't suppress
%them you get a simple image x and its blurred version b=Ax; e.g.
%[A,b,x] = blur(size(trues,1),20,2);
%figure; subplot(1,2,1); imagesc(reshape(x,size(trues,1),size(trues,1))); daspect([1 1 1]); 
%subplot(1,2,2); imagesc(reshape(b,size(trues,1),size(trues,1))); daspect([1 1 1]);

%We can now blur the image by multiplying all columns of clear image x with
%A
for i=1:3 
    b(:,i) = A*x(:,i); 
end

%and converting it into an image format for viewing
for i=1:3 
    bim(:,:,i) = reshape(b(:,i),size(trues,1),size(trues,2)); 
end

%We can now plot the true next to the blurry one for comparison
figure; subplot(1,2,1); imagesc(trues); daspect([1 1 1]); subplot(1,2,2); imagesc(bim); daspect([1 1 1]);

%We now start from the blurry bim (or b for computing) and want to go back
%to the true one trues. I will use an optimisation algorithm (primal-dual)
%for the Total Variation regularisation deblurring formulation. Each of the
%three tensor elements are processed individually. Note that I provide as
%input the same A I used above to blur the images. It also takes two other
%algorithmic parameters - 3rd and 4th arguement below - that need so
%"manual" tuning. The one that has the most impact on the image is the last
%one. I suggest leaving the other within 0.01 - 0.001.

%Before we run the deblurring algorithm we can check the "error" of the
%blurry image (since we know the exact image)
error_before = norm(x(:)-b(:))/norm(x(:));
%just so that we can compare with the restored image


for i=1:3
[imrest(:,:,i), output] = TVPrimDual(A, bim(:,:,i), 1e-2, 5e-5);
end

%A reduction in gradient norm by 3-4 orders of magnitude is sufficient to
%indicate that the algorithm have converged. Otherwise should try to vary
%the 4th argument. This could take some time to compute

%Reshape restored image for computing its error
for i=1:3 
    xrest(:,i) = reshape(imrest(:,:,i),size(trues,1)*size(trues,2),1); 
end

error_after = norm(x(:)-xrest(:))/norm(x(:));

%Plot for comparison
figure; 
subplot(1,3,1); imagesc(trues); daspect([1 1 1]);
subplot(1,3,2); imagesc(bim); daspect([1 1 1]); 
subplot(1,3,3); imagesc(imrest); daspect([1 1 1]);

%Suppose now that instead of knowing A as used to blur the image 
%(i.e. the camera's true PSF) we have an approximate one
[Awrong,~,~] = blur(size(trues,1),10,7);

%To see the difference between the true blurring kernel A and the asssumed
%one Awrong, we can plot some of their respective rows for comparison
figure; subplot(1,2,1); imagesc(reshape(A(135670,:),500,500)); daspect([1 1 1]);
subplot(1,2,2); imagesc(reshape(Awrong(135670,:),500,500)); daspect([1 1 1]);

%If we now run the image deblurring algorithm with this approximate blurring
%kernel we get
for i=1:3
[imrestw(:,:,i), output] = TVPrimDual(Awrong, bim(:,:,i), 1e-2, 5e-6);
end

%Reshape restored image for computing its error
for i=1:3 
    xrestw(:,i) = reshape(imrestw(:,:,i),size(trues,1)*size(trues,2),1); 
end

error = norm(x(:)-xrestw(:))/norm(x(:));

%Plot for comparison
figure; 
subplot(1,3,1); imagesc(trues); daspect([1 1 1]);
subplot(1,3,2); imagesc(imrest); daspect([1 1 1]); 
subplot(1,3,3); imagesc(imrestw); daspect([1 1 1]);
