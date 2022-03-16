function [rddata,U] = UFLPP(X,Y,dimension,sparsek,gamma,fuzzym,lemmda)

countloss=0;
Loss=[];
[data_row,data_col] =size(X);
if data_col>100
X=pca(X,100);
[data_row,data_col] =size(X);
end
cluster_n=length(unique(Y)); 
% Make sure data is zero mean
mapping.mean = mean(X, 1);
X = bsxfun(@minus, X, mapping.mean);
St=cov(X);
% Perform eigendecomposition of C
St(isnan(St)) = 0;
St(isinf(St)) = 0;
U=rand(cluster_n,data_row);
[sp,ps]=sort(U,1);
U(ps>sparsek)=0;
col_sum=sum(U);
U=U./col_sum;
mf = U.^fuzzym;
S=rand(data_row,data_row);
Scol_sum=sum(S);
S=S./Scol_sum;

maxgen =20;
for i = 1:maxgen
    %step 1 fixing w get smf and center
    center = mf*X./(repmat(sum(mf,2),1,data_col));
    dn=diag(sum(mf,1));
    dc=diag(sum(mf,2));
    D=diag(sum(S,2));
    L=D-0.5*(S+S');
    M=(X'*dn*X-2*X'*mf'*center+center'*dc*center+2*lemmda*X'*L*X);
    M=(M+M')/2;
    M(find(isnan(M)==1))= 0;
    G=chol(St,'lower');
    MM=inv(G)*M*(inv(G))';
    [W,B] = eig(MM);
    B(isnan(B)) = 0;
   [B, ind] = sort(diag(B));
    W = (inv(G))'*W;
    W = W(:,ind(1:dimension)); 
    ldata =real(X*W);
    lcenter =real(center*W);
    ldist =L2_distance_subfun(ldata',lcenter');
    distx = L2_distance_subfun((X*W)',(X*W)');

    Lossvalue=lemmda*sum(sum(S.*distx))+gamma*sum(sum(S.^2,2))+sum(sum(ldist'.*mf));
    if isnan(Lossvalue)
        break;
    end
    Loss=[Loss,Lossvalue];
       if countloss>1 
         if Loss(i-1)-Loss(i)<10^-5
                   break;
         end
     end
  
    [tmpp, ind] = sort(ldist,2);
    for j=1:size(ldist, 1)
        ldist(j, ind(j,(1 + sparsek):end)) = 0;
    end
    ldist(ldist ~= 0) = ldist(ldist ~= 0).^(-1/(fuzzym-1))';
    tmp=ldist';
    U=tmp./sum(tmp);
    mf = U.^fuzzym;
  
    mydis=L2_distance_subfun((X*W)',(X*W)');

    [temp, idx] = sort(mydis,2);
    A = zeros(data_row);
    islocal=1;
    for i=1:data_row
          if islocal == 1
            idxa0 = 1:data_row;
        else
            idxa0 = 1:data_row;
          end
         dxi = mydis(i,idxa0);
         ad = -(dxi*lemmda)/(2*gamma);
        A(i,idxa0) = EProjSimplex_new(ad,1);
    end
    S=(A+A')/2;
    countloss=countloss+1;
end
 rddata =real(X*W);
 
 %plot(Loss)
% heatlabel=S;
% heatlabel(heatlabel ~= 0)=heatlabel(heatlabel ~= 0)+0.5; 
% for j=1:size(heatlabel, 1)
%         heatlabel(j, j) = max(max(heatlabel));
% end  
%  [~,position]=sort(heatlabel,'descend');
%      for sorti=1:data_row
%          for sortj=1:data_row
%              if(sorti>12)
%                heatlabel(position(sorti,sortj),sortj)=0;
%              end
%          end
%      end
% heatmapHandle = heatmap( heatlabel, 'ColorMap', jet(100));
%  XLabels = 1:178;
% % Convert each number in the array into a string
% CustomXLabels = string(XLabels);
% % Replace all but the fifth elements by spaces
% CustomXLabels(mod(XLabels,10) ~= 0) = " ";
% % Set the 'XDisplayLabels' property of the heatmap 
% % object 'h' to the custom x-axis tick labels
% heatmapHandle.XDisplayLabels = CustomXLabels;
%  YLabels = 1:178;
% % Convert each number in the array into a string
% CustomYLabels = string(YLabels);
% % Replace all but the fifth elements by spaces
% CustomYLabels(mod(YLabels,10) ~= 0) = " ";
% % Set the 'XDisplayLabels' property of the heatmap 
% % object 'h' to the custom x-axis tick labels
% heatmapHandle.YDisplayLabels = CustomYLabels;
% 
end




