1. Install virtual environment
sudo apt install python-virtualenv
2. Then make an enviroment
virtualenv env -p python
3. Then activate env
source env/bin/activate
4. Then install requirements
pip install -r requirements.txt
5. Then run the website
python3 app.py
6. Finally website will be runnning on http://localhost:5000 

7. If there is tkinter error, then install
sudo apt-get install python3-tk 
