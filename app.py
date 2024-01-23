from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-my-app', methods=['POST'])
def run_my_app():
    result = subprocess.run(['./dist/SurfEncoder'], capture_output=True, text=True)
    return result.stdout

if __name__ == '__main__':
    app.run(debug=True, port=8080)
