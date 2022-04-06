'''

Flask_App

Author: Armin Berger
First created:  06/04/2022
Last edited:    06/04/2022

OVERVIEW:
This file seeks to deploy a pre-built ML model.
The user gives Text input to the model and the model then classifies whether
the news is reliable or not.

'''


from flask import Flask, render_template

app = Flask(__name__)


@app.route('/home')

def home():
    return render_template('home.html')

if __name__ == "__main__":
    app.run()
