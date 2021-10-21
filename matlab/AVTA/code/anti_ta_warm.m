function [inorout,p_prime,alpha_coe,dist]=anti_ta_warm(dataset,p,epsilon,alpha_0)

%%*********** Triangle algorithm using anti-pivot with warm up init
%Input: 
%   dataset: dim x K  matrix
%   p: Query point
%   epsilon : precision parameter
%   alpha_0: K x #of samples matrix
%Output:
%   inorout: 1 if p is in the convex hull, 0 other wise
%   p_prime: An approximation of p or witness of p
%   alpha_coe?K x #of samples matrix. Weight of each points using convex combination of colums of dataset
%   dist: The distance between p and p'
%


mat_a=dataset;

matA=mat_a;

[m,n]=size(matA);




diffmat=matA-repmat(p,1,n);

eudis=sqrt(diag(diffmat'*diffmat));

tmparr=find(eudis==min(eudis));
min_index=tmparr(1);
p_p=matA(:,min_index);
alpha=zeros(1,n);
alpha(min_index)=1;
if(length(alpha_0)==n && n>1)
   % size(alpha_0)
    %size(matA)
    [ral cal]=size(alpha_0);
    if ral>cal
        alpha_0=alpha_0';
    end
    alpha=alpha_0;
%     size(matA)
%     length(alpha)
    p_p=matA*alpha';
end


%disp('ok_3')
distance=min(eudis);
if(n<=1)
    inorout=0;
    p_prime=p_p;
    alpha_coe=alpha;
    dist=min(eudis);
    return;
end



iter=0;
iter_1=0;
iter_2=0;

inorout=1;

dist_vp=sum(diffmat.*diffmat,1);
beta_list=zeros(n,1);
while(sqrt((p-p_p)'*(p-p_p))>epsilon)
    found=0;
    distance=sqrt((p-p_p)'*(p-p_p));
    rnd_index=randperm(n,n);
    ppv=p_p(:,ones(1,n))-matA;
    norm2_ppv=sum(ppv.^2);
    norm_ppv=sqrt(norm2_ppv);
    gd=matA'*(p-p_p);
    p_norm=p'*p;
    p_p_norm=p_p'*p_p;
    dist_diff=(p_norm-p_p_norm)- 2*gd;
    index_pivot=find(dist_diff<=0);

    pre_beta_list=(p_p'*(p_p-p)+gd)./norm2_ppv';
    if length(index_pivot)==0
        found=0;
    else
        geq_index=find(pre_beta_list>=0);
        beta_list(geq_index)=min(1,pre_beta_list(geq_index));
        leq_index=find(pre_beta_list<0);

        beta_list(leq_index)=max(alpha(leq_index)'./(alpha(leq_index)-1)',pre_beta_list(leq_index));
        
        this_len=abs(beta_list).*norm_ppv';

        v_index=min(find(this_len==max(this_len)));
        


        beta=beta_list(v_index);
        
        alpha=(1-beta)*alpha;
        alpha(v_index)=alpha(v_index)+beta;
        p_p=(1-beta)*p_p+beta*matA(:,v_index);
        found=1;
    end
    if(found==0)
        inorout=0;
        break;
    end
end
p_prime=p_p;
alpha_coe=alpha;
dist=distance;
