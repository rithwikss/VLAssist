import cv2
import time
import threading
from queue import Queue
from image_processing import convert_frame_to_pil_image
from caption_generation import generate_caption
from response_generation import generate_response
from tts import speak_text

class VisionAssistant:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)  # Fixed property name
        
        self.current_caption = ""
        self.question_queue = Queue()
        self.lock = threading.Lock()
        self.running = True

    def process_question(self, question, caption):
        """Generate a response based on the user's question about the current scene."""
        prompt = f"""You are an AI assistant that answers questions about images based on captions.
        The current scene shows: {caption}
        
        User's question: {question}
        
        Please provide a clear and relevant answer based only on what's visible in the scene.
        If the question cannot be answered based on the visible content, say so politely."""

        response = generate_response(prompt, "", [])  # Modified to handle Q&A
        speak_text(response)
        return response

    def process_frame(self):
        """Continuously process frames to keep caption up to date."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            pil_image = convert_frame_to_pil_image(frame)
            caption = generate_caption(pil_image)
            
            with self.lock:
                self.current_caption = caption

            time.sleep(1)  # Update caption every second

    def handle_user_input(self):
        """Handle user questions from command line."""
        print("Ask questions about what you see (type 'quit' to exit):")
        while self.running:
            question = input("> ").strip()
            if question.lower() == 'quit':
                self.running = False
                break
            
            with self.lock:
                current_caption = self.current_caption
            
            self.process_question(question, current_caption)

    def display_frame(self):
        """Display the camera feed with current caption."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            with self.lock:
                caption = self.current_caption
            

            # Display the caption on frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Current scene: " + caption, (10, 30), 
                       font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            flipped_frame = cv2.flip(frame, 1)
            cv2.imshow('frame', flipped_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

    def run(self):
        """Run the assistant with all necessary threads."""
        # Start frame processing thread
        process_thread = threading.Thread(target=self.process_frame)
        process_thread.start()

        # Start display thread
        display_thread = threading.Thread(target=self.display_frame)
        display_thread.start()

        # Handle user input in the main thread
        self.handle_user_input()

        # Cleanup
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        process_thread.join()
        display_thread.join()

if __name__ == '__main__':
    assistant = VisionAssistant()
    assistant.run()