%%
%%%%%%% question 3 part 1 %%%%%%%%%%%
question3_helper2(1,2);
%%
question3_helper2(2,2);


%%
%%%%%%% question 3 part 2 %%%%%%%%%%%

clusters=[2,3,4,5,6,7,8,9,10,11,12];
l = length(clusters);
counters = zeros(l, 2);
%%
for image=1:2
    for cc = 1:l
        counters(cc,image)=question3_helper1(image,clusters(cc));
    end
end
%%
a=find(counters(:,1)==(max(counters(:,1))));
b=find(counters(:,2)==(max(counters(:,2))));
n1=clusters(a);
n2=clusters(b);
%%
n1=6;
n2=12;
%%
question3_helper2(1,n1);
question3_helper2(2,n2);
