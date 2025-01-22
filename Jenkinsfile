pipeline {
    agent any
    
    stages {
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
                
                sh 'docker exec fastapi-container python -m pytest tests/'
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