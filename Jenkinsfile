pipeline {
    agent any
    
    stages {
        stage('Create Static Directory') {
            steps {
                script {
                    sh 'mkdir -p static'
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Build Docker image
                    sh 'docker build -t fastapi-app .'
                }
            }
        }
        
        stage('Run Docker Container') {
            steps {
                script {
                    sh 'docker stop fastapi-container || true'
                    sh 'docker rm fastapi-container || true'
                    
                    sh 'docker run -d -p 8000:8000 --name fastapi-container fastapi-app'
                    
                }
            }
        }
        
        stage('Test') {
            steps {
                sh 'sleep 5'

                echo 'Running tests...'
                
            }
        }
    }
    
    post {
        failure {
            // Clean up container if pipeline fails
            sh 'docker stop fastapi-container || true'
            sh 'docker rm fastapi-container || true'
        }
    }
}