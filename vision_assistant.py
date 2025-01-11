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
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)
        
        self.current_caption = ""
        self.is_processing = False
        self.tts_queue = Queue()
        self.lock = threading.Lock()
        self.running = True

    def process_question(self, question, caption):
        """Generate a response based on the user's question about the current scene."""
        self.is_processing = True
        print("Processing your question...")
        
        prompt = f"""You are an AI assistant that answers questions about images based on captions.
        The current scene shows: {caption}
        
        User's question: {question}
        
        Please provide a brief, clear answer based only on what's visible in the scene.
        Keep responses concise - no more than 2 sentences.
        If the question cannot be answered based on the visible content, say so briefly."""

        try:
            response = generate_response(prompt, "", [])
            self.tts_queue.put(response)
            print("Response:", response)
        except Exception as e:
            print(f"Error generating response: {e}")
        finally:
            self.is_processing = False

    def handle_tts(self):
        """Handle text-to-speech in a separate thread."""
        while self.running:
            if not self.tts_queue.empty():
                text = self.tts_queue.get()
                speak_text(text)
            else:
                time.sleep(0.1)

    def process_frame(self):
        """Continuously process frames to keep caption up to date."""
        last_caption_time = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            current_time = time.time()
            # Update caption every 2 seconds to reduce processing load
            if current_time - last_caption_time >= 2:
                pil_image = convert_frame_to_pil_image(frame)
                caption = generate_caption(pil_image)
                
                with self.lock:
                    self.current_caption = caption
                last_caption_time = current_time

            time.sleep(0.1)

    def handle_user_input(self):
        """Handle user questions from command line."""
        print("Ask questions about what you see (type 'quit' to exit):")
        while self.running:
            if self.is_processing:
                print("Still processing previous question...")
                time.sleep(1)
                continue

            question = input("> ").strip()
            if question.lower() == 'quit':
                self.running = False
                break
            
            if not question:
                continue

            with self.lock:
                current_caption = self.current_caption
            
            # Process question in a separate thread
            threading.Thread(
                target=self.process_question,
                args=(question, current_caption)
            ).start()

            # Wait for the TTS to finish before accepting next input
            while not self.tts_queue.empty():
                time.sleep(0.1)

    def display_frame(self):
        """Display the camera feed with current caption and status."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            with self.lock:
                caption = self.current_caption

            # Display the caption and processing status
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Current scene: " + caption, (10, 30), 
                       font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display processing status
            if self.is_processing:
                cv2.putText(frame, "Processing question...", (10, 60),
                           font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
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

        # Start TTS handling thread
        tts_thread = threading.Thread(target=self.handle_tts)
        tts_thread.start()

        # Handle user input in the main thread
        self.handle_user_input()

        # Cleanup
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        process_thread.join()
        display_thread.join()
        tts_thread.join()

if __name__ == '__main__':
    assistant = VisionAssistant()
    assistant.run()