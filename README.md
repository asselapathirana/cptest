 This app uses dokku:storage 

https://github.com/dokku/dokku/blob/master/docs/advanced-usage/persistent-storage.md

data directory should have :
concrete_best.model                                                                   
concrete_lb.pickle 
 
 scp -r data/* root@s.wa.pathirana.net:/opt/dokku/cp/data/.
(then ssh dokku@s.wa.pathirana.net ps:restart cp 
 
 # dokku storage:report cp
=====> cp storage information
       Storage build mounts:
       Storage deploy mounts:         -v /opt/dokku/cp/data:/app/data
       Storage run mounts:            -v /opt/dokku/cp/data:/app/data
