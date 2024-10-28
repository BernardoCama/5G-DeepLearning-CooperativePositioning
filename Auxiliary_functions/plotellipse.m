function plotEllipse(AP,ellipsePoints,UE)
x_ell = ellipsePoints(1,:);
y_ell = ellipsePoints(2,:);

fig = figure(1); 
hold on
%fig.WindowState = 'maximized';
plot( AP(:,1) , AP(:,2) , '^','MarkerSize',10,'MarkerEdgeColor',[147,0,0]./255,'MarkerFaceColor',[147,0,0]./255)
fill(x_ell,y_ell,[.9 .95 1],'edgecolor',[0, 0.4470, 0.7410],'linewidth',2);alpha(.5)
plot(UE(1),UE(2),'pk','LineWidth',1,'MarkerSize',9,'MarkerFaceColor',[.4 .4 1]);
plot([AP(:,1),repmat(UE(1),size(AP,1),1)]',[AP(:,2),repmat(UE(2),size(AP,1),1)]','--','LineWidth',1,'color',[220,220,220]./255)
legend('AP','Ellipse','UE','location','best')
xlabel('[m]'), ylabel('[m]');
grid on
xlim([min(AP(:,1), [], 'all') max(AP(:,1), [], 'all')])
ylim([min(AP(:,2), [], 'all') max(AP(:,2), [], 'all')])
% axis equal


end