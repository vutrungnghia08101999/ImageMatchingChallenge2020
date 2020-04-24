rm *.yml
rm *.py
rm *.ipynb
rm *.txt
rsync -avz STANFORD:~/workspace/hardnet/*.* ./

