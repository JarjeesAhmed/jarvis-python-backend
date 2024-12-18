from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/run-project', methods=['POST'])
def run_project():
    print("Route /run-project hit")
    try:
        script_path = 'F:\\Jarjees Work\\my-projects\\jarvis-python-backend\\image_script.py'
        print(f"Running script: {script_path}")
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        print(f"Script output: {result.stdout}")
        if result.returncode != 0:
            print(f"Script error: {result.stderr}")
            return jsonify({'success': False, 'error': result.stderr}), 500
        return jsonify({'success': True, 'output': 'test'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
