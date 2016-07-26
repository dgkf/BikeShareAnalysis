from flask import Flask, redirect, render_template

app = Flask(__name__)

@app.route('/')
def main():
    return redirect('/index')

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=33507)
