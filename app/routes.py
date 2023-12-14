from flask import Flask, render_template, url_for, request
import model.pipeline as pipeline # model\pipeline.py
import videogen as videogen
import slidegen as slidegen
import audiogen as audiogen
import drive_interaction as drive_interaction

import parsing as parsing
import datetime

import shutil
import os

import json
from flask import jsonify
from app import app
from flask import Response

def execute_pipeline(document):

	if os.path.exists('images'):
		shutil.rmtree('images')
	if os.path.exists('output'):
		shutil.rmtree('output')
	
	os.mkdir('output')
	document['no_of_slides'], document['slides'] = pipeline.get_slide_content(document['text'])

	# print (document)
	#mutithreading can be used here
	slidegen.create_slides(document)
	audiogen.synthesize_audio(document)

	number_of_slides = document['no_of_slides']+2 #the first two slides
	videogen.generate_video(number_of_slides)

	shutil.rmtree('output', ignore_errors=True)
	shutil.rmtree('images', ignore_errors=True)

@app.before_request
def basic_authentication():
    if request.method.lower() == 'options':
        return Response()

@app.route('/')
def home():
	return jsonify({'message': 'Welcome to the SlideIt-API'})


@app.route('/predict_text', methods=['POST', 'GET'])
def predict_text():

	if request.method == 'POST':
		# extract the prediction from the model
		request_data = json.loads(request.data.decode('utf-8'))
		raw_data = request_data['data']
		document = parsing.parse_text(raw_data)
		execute_pipeline(document)
		file_link = drive_interaction.uploadFiles()
		return jsonify({'message': "you now get the pdf and output video", "link": file_link})

	if request.method == 'GET':
		return jsonify({'message': 'Please use the POST method'})	

@app.route('/predict_url', methods=['POST', 'GET'])
def predict_url():

	if request.method == 'POST':
		# extract the prediction from the model
		request_data = json.loads(request.data.decode('utf-8'))
		raw_data = request_data['url']
		document = parsing.parse_url(raw_data)
		execute_pipeline(document)				
		file_link = drive_interaction.uploadFiles()
		return jsonify({'message': "you now get the pdf and output video", "link": file_link})

	if request.method == 'GET':
		return jsonify({'message': 'Please use the POST method'})

@app.route('/predict_upload', methods=['POST', 'GET'])
def predict_upload():

	if request.method == 'POST':
		# extract the prediction from the model
		request_data = json.loads(request.data.decode('utf-8'))
		raw_data = request_data['upload']
		document = parsing.parse_upload(raw_data)
		execute_pipeline(document)
		file_link = drive_interaction.uploadFiles()
		return jsonify({'message': "you now get the pdf and output video", "link": file_link})

	if request.method == 'GET':
		return jsonify({'message': 'Please use the POST method'})	