% Read files mat1.mat through mat20.mat, file1.txt through file20.txt, 
% and image1.jpg through image20.jpg.  Files are in the current directory.
% Use fullfile() if you need to prepend some other folder to the base file name.
% Adapt to use which ever cases you need.
GREAT_IMS = {};

for k = 1:10000
	% Create an image filename, and read it in to a variable called imageData.
	jpgFileName = strcat('./Small_Galaxy_Zoo/image-', num2str(k), '.jpg');
	if exist(jpgFileName, 'file')
		imageData = imread(jpgFileName);
        GREAT_IMS{k} = rgb2gray(imageData);
	else
		fprintf('File %s does not exist.\n', jpgFileName);
    end
end

save('GREAT_IMS30.mat','GREAT_IMS');