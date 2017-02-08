%% PROJET RESEAUX DE NEURONES
%
% Author : Mauricio Caceres
% Version : RN_STANDART_V0
% Comments : add optimisation and temps mesure

%% TODO checklist
%=================================================
% - add temps mesure --DONE
% - add optmissations
% -o Shuffle the training set so that successive training examples never (rarely) belong the same class
% o Present input examples with nearly equal frequencies per classes (ça c’est fait !)
% o Entropie : Present input examples that produce a large error more frequently than
%           examples that produce a small error. However, one must be careful when
%           perturbating the normal frequencies of input samples
% o Shuffle the training set so that successive training examples never (rarely) belong to
%           the same class
% o Entropie : Present input examples that produce a large error more frequently than
%         examples that produce a small error. However, one must be careful when
%         perturbating the normal frequencies of input samples

%=================================================
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
display('')
display('//=========================== RS 3================================//')
display('// MLP : mu variable, initialisation de poids selon distr. normal//')
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
%% CALCUL DES ATTRIBUTS
%donnees sans normalisation
 % figure(1)
 % plot(attributs);figure(gcf);
 % title('attributs non normalises')
%donn�es normalise
for ligne = 1:size(attributs,1)
    ecart = sqrt(var(attributs(ligne,:)));
    attributs(ligne,:) = (attributs(ligne,:)-mean(attributs(ligne,:)))/ecart;
end
% figure(2)
 % plot(attributs);figure(gcf);
 % title('attributs non normalises')
% close all


sonsUn = (1:3:60);
sonsDeux = (2:3:60);
sonsTrois = (3:3:60);

t = zeros(3*20,1);
t(1:3:end) = 1+3*(randperm(20) -1);
t(2:3:end) = 2+3*(randperm(20) -1);
t(3:3:end) = 3+3*(randperm(20) -1);


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

%% --------------------
%% OTHERS PARAMETERS
%% --------------------

L_in = 4097+1;%on rajoute le bias
L_cachee = 100+1;%on rajoute le bias
L_out = 3;

%Init parametres de distribution de proba. poids
sigma1=1/sqrt(L_in);
sigma2 = 1/sqrt(L_cachee);

%creation the matrices de poids
C = normrnd (0, sigma1, L_in, L_cachee);
W =   normrnd(0, sigma2, L_cachee, L_out);
C_init=C;
W_init= W;
ym = zeros(3,1);

%Init taux d'apprentissage
alpha1 = .5;
alpha2 = 0.3;
muo = 0.6
mu1 = muo/(1+alpha1*1);
mu2 = muo/(1+alpha2*1);

tt = zeros(15,1);

tt(1:3:end) = 1+3*(randperm(5) -1);
tt(2:3:end) = 2+3*(randperm(5) -1);
tt(3:3:end) = 3+3*(randperm(5) -1);

%% ------------------------------
%
%  ALGORITHME GENERAL
%
%% ------------------------------

%general variables
epsilon=zeros(1,60); %erreur en chaque iteration
input = [ones(1,60); attributs];
e = 1e-4;
iter =1;
boucle=1;
i = 1;
more off;




tic; % init counter time
%% --------------------
%  ITERATIONS LOOP
%% --------------------
while(boucle==1)
    %reorganisation des examples
    t(1:3:end) = 1+3*(randperm(20) -1);
    t(2:3:end) = 2+3*(randperm(20) -1);
    t(3:3:end) = 3+3*(randperm(20) -1);
    %reorganisation examples de test
    tt(1:3:end) = 1+3*(randperm(5) -1);
    tt(2:3:end) = 2+3*(randperm(5) -1);
    tt(3:3:end) = 3+3*(randperm(5) -1);

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

         em = ym - y;
        epsilon(i) = (0.5/L_out)*em'*em;
        global_error_evolution(iter) = epsilon(i);

        % Algorihme de mise � jour des poids de sortie
        deltam_out = em.*gprime(zm);
        for neurone = 1: L_out
            W(:,neurone) = W(:,neurone) + ((mu1/L_out)*(deltam_out(neurone)*r));
        end

        % Algorihme de mise � jour des poids d'entr�e

        deltaj_out=zeros(L_cachee,1 );
        for neurone = 1:L_cachee
           somme_delta_w(neurone) =  W(neurone,:)*deltam_out;

        end

        for neurone = 1:L_cachee
          deltaj_out(neurone) = gprime(vj(neurone))*somme_delta_w(neurone);
          C(:,neurone) = C(:,neurone) + (mu2/L_out).*input(:,t(i))*deltaj_out(neurone);
        end
    end %fin boucle de presentation d'examples

        %update mu
        mu1 = muo/(1+alpha1*i);
        mu2 = muo/(1+alpha2*i);
        storemu(i)=mu1; %store for graphs
        storemu2(i) = mu2;
        %calcule eqm pour phase de training
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

    %parametre critere pour la fin de l'ALGORITHME
    if(error<=e)
          boucle = 0;
    end
      if(error<=0.001)
          alpha = 0.4;
      end
      if(error<=0.0001)
          alpha = 0.5;
      end
      if(error<=0.00002)
      alpha = 0.9;
      end

    iter = iter +1; %counter associate to the while loop
end
elapsed_time = toc



%% --------------------
%  TRAIN GRAPHS
%% --------------------
figure(5)
plot(eqm,'-b','LineWidth',2);
hold on;
plot(eqm_test,'-r','LineWidth',2);
grid ();
title('Error EQM in training and test phase','FontSize',12,'FontName','5');
xlabel('iterations','FontSize',12);
ylabel('Error value','FontSize',12);
legend('EQM train','EQM test');

figure(4)
plot(storemu1,'LineWidth',2)
grid()
hold on;
plot(storemu2,'LineWidth',2)
title('Iteration error global evolution (error in each example) ','FontSize',12);
xlabel('iterations','FontSize',12);
ylabel('Error','FontSize',12);

figure(6)
subplot (2, 1, 1)
plot(W_init,'-o','LineWidth',2)
axis ([0 120 -0.8 0.8])
grid()
title('Initial Weights (Normal distribution initialisation) ','FontSize',12);
xlabel('Weight id','FontSize',12);
ylabel('Weight value','FontSize',12);
subplot (2, 1, 2)
plot(W,'-o','LineWidth',2)
axis ([0 120 -0.8 0.8])
grid()
title('Final Weights ','FontSize',12);
xlabel('Weight id','FontSize',12);
ylabel('Weight value','FontSize',12);

%% --------------------
%  TEST GRAPHs
%% --------------------
figure(7)
plot(global_error_evolution,'LineWidth',2)
grid()
title('Iteration error global evolution (error in each example) ','FontSize',12);
xlabel('iterations','FontSize',12);
ylabel('Error','FontSize',12);
%% FIN


%% DOCUMENTATION
% sauvegarde les images pour le rapport
h = get(0,'children');
for i=length(h):-1:1
  saveas(h(i), ['rs4' num2str(length(h)+1-i)], 'png');
end
