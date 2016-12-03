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

%% Exercise 2 - S�lection des attibuts de la base de donn�s d'apprentissage
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
%donn�es sans normalisation
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

L_in = 4097+1;%on rajoute le bias
L_cachee = 100+1;%on rajoute le bias
L_out = 3;

mu = 0.4 %taux d'apprentissage
sigma1=1/L_in;
sigma2 = 1/L_cachee;
C = lognrnd (mu, sigma, L_in, L_cachee) %[ a + (b-a).*rand(L_in,L_cachee)];
W = lognrnd (mu, sigma, L_cachee, L_out)%[a + (b-a).*rand(L_cachee,L_out)];
ym = zeros(3,1);

%% ALGORITHME GENERALs
%close all
epsilon=zeros(1,60);

input = [ones(1,60); attributs];
e = 1e-4;
iter =1;
boucle=1;

%pause
i = 1;
more off;

t1 = tic;
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
        global_error_evolution(iter) = epsilon(i);
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
    %je calcule un "error" pour l'afficher aussi
    error = sum(epsilon)/60
    %fprintf('error : %f\n', error);
    %fflush(stdout);

    %parametre critere pour la fin de l'ALGORITHME
    eqm(iter) = error;
    iter = iter +1; %counter associate to the while loop
    if(error<=e)
        boucle = 0;
    end
end
t2 = toc;
temp = t2 - t1

%% calcul de Erreur quadratique moyenne
figure(3)
plot(eqm,'-b','LineWidth',2);
title('Error ','FontSize',12);
xlabel('iterations','FontSize',12);
ylabel('Error','FontSize',12);

figure(4)
plot(global_error_evolution,'LineWidth',2)
title('Iteration error global evolution (error in each example) ','FontSize',12);
xlabel('iterations','FontSize',12);
ylabel('Error','FontSize',12);


%% DOCUMENTATION



%% FIN
