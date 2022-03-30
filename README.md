 This app uses dokku:storage 

https://github.com/dokku/dokku/blob/master/docs/advanced-usage/persistent-storage.md

data directory should have :
concrete_best.model                                                                   
concrete_lb.pickle 


 scp -r data/* root@s.wa.pathirana.net:/opt/dokku/cp/data/.
(then ssh dokku@s.wa.pathirana.net ps:restart cp 
 
 dokku storage:mount cp /opt/dokku/cp/data:/app/data

 # dokku storage:report cp
=====> cp storage information
       Storage build mounts:
       Storage deploy mounts:         -v /opt/dokku/cp/data:/app/data
       Storage run mounts:            -v /opt/dokku/cp/data:/app/data


Note: libGL.so.1 will be required for opencv package. Do the following
sudo dokku plugin:install https://github.com/dokku-community/dokku-apt apt
there need to be a file apt-packages with
libgl1-mesa-glx

then when the app is git pushed to dokku, the packge will be automatically installed. 


letsencyrpt
dokku letsencrypt:enable cp
