function [] = Main_23_02_2023()
clc;
clear all;
close all;
warning off;

global Train_Data Train_Target Test_Data Test_Target Bestsol Images groundtruth

%% Read Dataset
an = 0;
if an == 1
    Directory = './Dataset/';
    File_List= dir(Directory);
    iteration = 1;
    for i = 1:length(File_List) - 2
        sub_folder = strcat(Directory, File_List(i + 2).name, '/');
        filesplit = split(File_List(i + 2).name,'_');
        variety = filesplit(end);
        Image_List = dir(sub_folder);
        for j = 1:length(Image_List) - 2
            disp(strcat(num2str(i), '-', num2str(j)));
            filename = strcat(sub_folder, Image_List(j + 2).name);
            image = imread(filename);
            image = rgb2gray(image);
            image = imresize(image, [128 128]);
            Images{iteration} = image;
            if strcmp(variety, 'healthy')
                Target(iteration) = 0;
            else
                Target(iteration) = 1;
            end
            iteration = iteration + 1;
        end
    end
    Target = Target';
    save Images Images
    save Target Target
end


%% Optimization for Segmentation
an = 0;
if an == 1
    load Images
    load groundtruth
    Npop = 10;
    Ch_len = 3;  % Hidden Neuron Count, Epoch, Step per Epoch
    xmin = repmat([5, 5, 300], Npop, 1);  % Minimum Bound
    xmax = repmat([255, 50, 1000], Npop, 1);  % maxiumum Bound
    initsol = xmin + (xmax - xmin) .* rand(Npop, Ch_len);                       % Initial Solution
    fname = 'obj_fun';
    itermax = 50;
    
    disp('DOX');
    [bestfit,fitness,bestsol,time] = DOX(initsol,fname,xmin,xmax,itermax); % DOX
    Do.bf = bestfit; Do.fit = fitness; Do.bs = bestsol; Do.ct = time;
    save Do Do
    
    disp('EOO');
    [bestfit,fitness,bestsol,time] = EOO(initsol,fname,xmin,xmax,itermax); % EOO
    Eoo.bf = bestfit; Eoo.fit = fitness; Eoo.bs = bestsol; Eoo.ct = time;
    save Eoo Eoo
    
    disp('RSA');
    [bestfit,fitness,bestsol,time] = RSA(initsol,fname,xmin,xmax,itermax); % RSA
    Rsa.bf = bestfit; Rsa.fit = fitness; Rsa.bs = bestsol; Rsa.ct = time;
    save Rsa Rsa
    
    disp('TFMOA');
    [bestfit,fitness,bestsol,time] = TFMOA(initsol,fname,xmin,xmax,itermax);  % TFMOA
    Tfmo.bf = bestfit; Tfmo.fit = fitness; Tfmo.bs = bestsol; Tfmo.ct = time;
    save Tfmo Tfmo
    
    disp('Proposed');
    [bestfit,fitness,bestsol,time] = Proposed(initsol,fname,xmin,xmax,itermax);  % PROPOSED
    Prop.bf = bestfit; Prop.fit = fitness; Prop.bs = bestsol; Prop.ct = time;
    save Prop Prop
end


%% Segmentation
an = 0;
if an == 1
    Algm = {'Do', 'Eoo', 'Rsa', 'Tfmoa', 'Prop'};
    load Images
    load groundtruth
    for n = 1:length(Algm)
        for i = 1: length(Images)
            disp(strcat(num2str(n), '-', num2str(i)));
            eval(['load ', char(Algm(n))])
            eval(['sol = ', char(Algm(n)), '.bs;']);
            [Seg_Img, Eval_all_Dice] = Model_AMST_ASPP();
        end
    end
    save Seg_Img Seg_Img
    save Eval_all_Dice Eval_all_Dice
end


%% Optimization for Classification
an = 0;
if an == 1
    load Seg_Img
    load Target
    Feat = Seg_Img;
    tar = Target; 

    Npop = 10; % Population size
    Ch_len = 6;  
    xmin = [5, 5, 1, 5, 5, 1] .* ones(Npop, Ch_len);  % Minimum Bound
    xmax = [255, 50, 4, 255, 50, 4] .* ones(Npop, Ch_len);   % maxiumum Bound
    initsol = unifrnd(xmin,xmax);
    itermax = 50;
    fname = 'Objfun_cls';

    learnperc = round(size(Feat, 1) * 0.75);
    Train_Data = Feat(1:learnperc, :);
    Test_Data = Feat(learnperc + 1:end, :);
    Train_Target = tar(1:learnperc);
    Test_Target = tar(learnperc + 1:end);

    disp('DO');
    [bestfit,fitness,bestsol,time] = DOX(initsol,fname,xmin,xmax,itermax); % DOX
    Do1.bf = bestfit; Do1.fit = fitness; Do1.bs = bestsol; Do1.ct = time;
    save Do1 Do1

    disp('EOO');
    [bestfit,fitness,bestsol,time] = EOO(initsol,fname,xmin,xmax,itermax); % EOO
    Eoo1.bf = bestfit; Eoo1.fit = fitness; Eoo1.bs = bestsol; Eoo1.ct = time;
    save Eoo1 Eoo1

    disp('RSA');
    [bestfit,fitness,bestsol,time] = RSA(initsol,fname,xmin,xmax,itermax); % RSA
    Rsa1.bf = bestfit; Rsa1.fit = fitness; Rsa1.bs = bestsol; Rsa1.ct = time;
    save Rsa1 Rsa1

    disp('TFMO');
    [bestfit,fitness,bestsol,time] = TFMO(initsol,fname,xmin,xmax,itermax);  % TFMO
    Tfmo1.bf = bestfit; Tfmo1.fit = fitness; Tfmo1.bs = bestsol; Tfmo1.ct = time;
    save Tfmo1 Tfmo1

    disp('Proposed');
    [bestfit,fitness,bestsol,time] = Proposed(initsol,fname,xmin,xmax,itermax);  % PROPOSED
    Prop1.bf = bestfit; Prop1.fit = fitness; Prop1.bs = bestsol; Prop1.ct = time;
    save Prop1 Prop1
end

%% Classification
an = 0;
if an == 1
    load Feature
    load Target
    Algm = {'Do1', 'Eoo1', 'Rsa1', 'Tfmo1', 'Prop1'};
    learnper = [0.35 0.45 0.55 0.65 0.75 0.85];
    net_in = [];
    for i = 1:length(learnper)
        learnperc = round(length(Target) * learnper(i));
        for j = 1:length(Algm)
            Feat = Feature;
            Train_Data = Feat(1:learnperc, :);
            Train_Target = Target(1:learnperc);
            Test_Data = Feat(learnperc + 1:end, :);
            Test_Target = Target(learnperc + 1:end);
        end
        
        [Eval(6, :), net_out{i, 6}] = Model_CNN(net_in);
        [Eval(7, :), net_out{i, 7}] = Model_RAN(net_in);
        [Eval(8, :), net_out{i, 8}] = Model_EL(net_in);
        sol = Bestsol;
        Bestsol = 20;
        [Eval(9, :)] = Model_MSDHAN(net_in);
        [Eval(10, :)] = Model_MSDHAN(net_in);
        Eval_all{i} = Eval;
    end
    save Eval_all Eval_all
end
Plot_Results()

end