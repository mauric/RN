%% PROJET RESEAUX DE NEURONES
%
% Author : Mauricio Caceres
% Version : RN_STANDART_V0
% Comments : fonctionnement normal du reseux, sans aucun optimisation

%% TODO checklist
%
% You can fork this projet in
%
%% https://github.com/mauric/RN.git

clc
clear all
close all
display('//================================================================//')
display('//                    RECHERCHE OPERATIONNELLE                    //')
display('//================================================================//')

%% --------------------
%% TRAINING DATA BASE
%% --------------------

in = 1;
attributs = zeros(4097,60); % TODO change this init set, calculate it!
Nfft = 8912;
for numfich = 1:20
    for typeson = 1:3
        name = ['BDD_',num2str(typeson),'_',num2str(numfich),'.wav']; %'BDD_2_1.wav'
        [data,fs,Nbits] = wavread(name);%lecture de fichier wav
        % calcul des attributs
        L = size(data,1);
        X = fft(data.*hamming(L),Nfft);
        Sxx = 1/Nfft*abs(X).^2;
        attributs(:,in) = Sxx(1:size(attributs,1));
        in=in+1;
    end
end
attributs_no_normalise = attributs;
for ligne = 1:size(attributs,1)
    ecart = sqrt(var(attributs(ligne,:)));
    attributs(ligne,:) = (attributs(ligne,:)-mean(attributs(ligne,:)))/ecart;
end

%% --------------------
%% TEST DATA BASE
%% --------------------

%read data base
in = 1;
data_test = zeros(4097,15);
Nfft = 8912;
for numfich = 1:5
    for typeson = 1:3
        name = ['TEST_',num2str(typeson),'_',num2str(numfich),'.wav']; %'TEST_2_1.wav'
        [dataTest,fs,Nbits] = wavread(name);%lecture de fichier wav
        % calcul des data_test
        L = size(dataTest,1);
        X = fft(dataTest.*hamming(L),Nfft);
        Sxx = 1/Nfft*abs(X).^2;
        data_test(:,in) = Sxx(1:size(data_test,1));
        in=in+1;
    end
end

%normalise data
for ligne = 1:size(data_test,1)
    ecart = sqrt(var(data_test(ligne,:)));
    data_test(ligne,:) = (data_test(ligne,:)-mean(data_test(ligne,:)))/ecart;
end
epsilon_test=zeros(1,15); %erreur pour chaque example presenté
input_test = [ones(1,15); data_test];

sonsUn = (1:3:60);
sonsDeux = (2:3:60);
sonsTrois = (3:3:60);

t = zeros(3*20,1);
t(1:3:end) = 1+3*(randperm(20) -1);
t(2:3:end) = 2+3*(randperm(20) -1);
t(3:3:end) = 3+3*(randperm(20) -1);

L_in = 4097+1;%on rajoute le bias
L_cachee = 100+1;%on rajoute le bias
L_out = 3;
mu = 0.5 %taux d'apprentissage
a=-0.5;
b = 0.5;
C = [ a + (b-a).*rand(L_in,L_cachee)];
W = [a + (b-a).*rand(L_cachee,L_out)];

ym = zeros(3,1);
C_init=C;
W_init= W;
%% ALGORITHME GENERALs
%close all
epsilon=zeros(1,60);

input = [ones(1,60); attributs];
e = 1e-4;
iter =1;
boucle=1;


tt = zeros(15,1);

tt(1:3:end) = 1+3*(randperm(5) -1);
tt(2:3:end) = 2+3*(randperm(5) -1);
tt(3:3:end) = 3+3*(randperm(5) -1);


i = 1;
more off;
while(boucle==1)
    %reorganisation des examples
    t(1:3:end) = 1+3*(randperm(20) -1);
    t(2:3:end) = 2+3*(randperm(20) -1);
    t(3:3:end) = 3+3*(randperm(20) -1);

    for i = 1:60

        %% calcul de sortie de couchee cachee
        vj = C'*input(:,t(i));
        r =sigmoide(vj);
        r(1)=-1;%set a bias
        %% calcul de sortie de couchee sortie
        zm = W'*r;
        % pause
        y = sigmoide(zm);

        %% calcul de l'erreur
        typeson = rem(t(i),3);
        verif_typeson(i) = typeson; %print this to verified  typeson
        if typeson == 0
            ym = [0 0 1]';
        elseif typeson==1
            ym = [1 0 0]';
         else
            ym = [0 1 0]';
        end

        %  em = y - ym;
         em = ym - y;
        epsilon(i) = (0.5/L_out)*em'*em;
        % fprintf('error : %f\n', epsilon(i));
        % Algorihme de mise � jour des poids de sortie
        deltam_out = em.*gprime(zm);
        for neurone = 1: L_out
            W(:,neurone) = W(:,neurone) + ((mu/L_out)*(deltam_out(neurone)*r));
        end

        % Algorihme de mise � jour des poids d'entr�e

        deltaj_out=zeros(L_cachee,1 );
        for neurone = 1:L_cachee
           somme_delta_w(neurone) =  W(neurone,:)*deltam_out;

        end

        for neurone = 1:L_cachee
          deltaj_out(neurone) = gprime(vj(neurone))*somme_delta_w(neurone);
          C(:,neurone) = C(:,neurone) + (mu/L_out).*input(:,t(i))*deltaj_out(neurone);
        end
 %pause
    end
    error = sum(epsilon)/60
    eqm(iter) = error;

    %% --------------------
    %  TEST LOOP
    %% --------------------
    for i = 1:15
        %% calcul de sortie de couchee cachee
        vj_test = C'*input_test(:,tt(i));
        r_test =sigmoide(vj_test);
        r_test(1)=-1;%set a bias
        %% calcul de sortie de couchee sortie
        zm_test = W'*r_test;
        % pause
        y_test = sigmoide(zm_test);

        %% calcul de l'erreur
        typeson = rem(tt(i),3);
        if typeson == 0
            ym_test = [0 0 1]';
        elseif typeson==1
            ym_test = [1 0 0]';
         else
            ym_test = [0 1 0]';
        end
        em_test = ym_test - y_test;
        epsilon_test(i) = (0.5/L_out)*em_test'*em_test;
        global_error_evolution_test(iter) = epsilon_test(i);

    end
    %je calcule un "error" pour l'afficher aussi
    error_test = sum(epsilon_test)/15
    eqm_test(iter) = error_test;
    epsilon_test=zeros(1,15);

    %reset epsilon
    epsilon=zeros(1,60);
    epsilon_test=zeros(1,60);


    iter = iter +1;
    if(error<=e)
        boucle = 0;
    end
end




%% --------------------
%  TRAIN GRAPHS
%% --------------------
figure(1)
plot(eqm,'-b','LineWidth',2);
hold on;
plot(eqm_test,'-r','LineWidth',2);
grid ();
title('Error EQM in training and test phase','FontSize',12,'FontName','5');
xlabel('iterations','FontSize',12);
ylabel('Error value','FontSize',12);
legend('EQM train','EQM test');


figure(2)
subplot (2, 1, 1)
plot(W_init,'-o','LineWidth',2)
axis ([0 120 -0.8 0.8])
grid()
title('Initial Weights ','FontSize',12);
xlabel('Weight id','FontSize',12);
ylabel('Weight value','FontSize',12);
subplot (2, 1, 2)
plot(W,'-o','LineWidth',2)
axis ([0 120 -0.8 0.8])
grid()
title('Final Weights ','FontSize',12);
xlabel('Weight id','FontSize',12);
ylabel('Weight value','FontSize',12);

figure(3)
subplot (2, 1, 1)
hist (W_init(:,1), 25,'facecolor', 'r', 'edgecolor', 'b');
colormap (summer ());
grid()
title('Initial Weights','FontSize',12);
xlabel('Weight id','FontSize',12);
ylabel('Weight value','FontSize',12);
subplot (2, 1, 2)
hist (W(:,1), 25, 'facecolor', 'r', 'edgecolor', 'b');
colormap (summer ());
grid()
title('Final Weights ','FontSize',12);
xlabel('valuer','FontSize',12);



figure(4)
subplot (2, 1, 1)
hist (C_init(:,1), 25, 'facecolor', 'r', 'edgecolor', 'b');
colormap (summer ());
grid()
title('Initial Weights ','FontSize',12);
xlabel('Weight id','FontSize',12);
ylabel('Weight value','FontSize',12);
subplot (2, 1, 2)
hist (C(:,1), 25, 'facecolor', 'r', 'edgecolor', 'b');
colormap (summer ());
grid()
title('Final Weights ','FontSize',12);
xlabel('valuer','FontSize',12);





%% --------------------
%  TEST GRAPHs
%% --------------------

%% --------------------
%  DATA BASE GRAPH
%% --------------------

 figure(5)
 subplot (2, 1, 1)
 plot(attributs(:,1:5));
 axis ([0 120 -1.5 8])
 grid()
 title('attributs normalises ','FontSize',12);
 xlabel('component','FontSize',12);
 ylabel('attributs value','FontSize',12);
 subplot (2, 1, 2)
 plot(attributs_no_normalise(:,1:5));
 axis ([0 120 -0.5 15])
 grid()
 title('Attributs non normalises ','FontSize',12);
 xlabel('component','FontSize',12);
 ylabel('attributs value','FontSize',12);





 %% --------------------
 %% DOCUMENTATION
 %% --------------------
 % sauvegarde les images pour le rapport
 h = get(0,'children');
 for i=length(h):-1:1
   saveas(h(i), ['rs' num2str(length(h)+1-i)], 'png');
 end
