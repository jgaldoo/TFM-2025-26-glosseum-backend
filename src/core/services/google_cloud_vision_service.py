from google.oauth2 import service_account
from google.cloud.vision_v1 import ImageAnnotatorClient, AnnotateImageRequest, AnnotateImageResponse
from google.cloud.vision_v1.types import Image, Feature

from src.core.config import settings
from src.core.utils.text_utils import clean_quoting

"""Obtains paragraphs from a given full text annotation.

    Parses the content of a full text annotation to obtain the paragraphs in the OCR extracted text.

    Args:
        full_text_annotation: The full_text_annotation item from an AnnotateImageResponse.

    Returns:
        list: The list of extracted paragraphs. Can be empty.
    """
def get_text_paragraphs(full_text_annotation):
    paragraph_content = []

    for page in full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:

                words = []
                skip_iteration = False

                for i, word in enumerate(paragraph.words):
                    if not skip_iteration:
                        current_word = "".join(symbol.text for symbol in word.symbols)

                        # Join hyphen separated line broken words
                        if current_word != "-" and current_word.endswith("-") and i + 1 < len(paragraph.words):
                            skip_iteration = True

                            next_word = "".join(symbol.text for symbol in paragraph.words[i + 1].symbols)
                            words.append(current_word[:-1] + next_word)
                        else:
                            words.append(current_word)
                    else:
                        skip_iteration = False


                paragraph_content.append(" ".join(clean_quoting(words)))

    return paragraph_content


class VisionService:
    def __init__(self):
        self.client = ImageAnnotatorClient(
            credentials=service_account.Credentials.from_service_account_file(
                settings.google_cloud_vision_api_credentials
            ),
            client_options={"api_endpoint": "eu-vision.googleapis.com"}
        )

    """Extracts text from a given image.
    
    Calls the associated Google Cloud Vision API endpoint to detect text in an image
    and extract it through OCR.
    
    Args:
        image_content: The raw bytes of the image to process.
        
    Returns:
        list | None: The list of extracted text annotations, or None if no text was extracted.
    """
    def detect_text(self, image_content):

        # Despite IDE expecting dicts, AnnotateImageRequest accepts protobuf format
        request = AnnotateImageRequest(
            image=Image(content=image_content),
            features=[Feature(type=Feature.Type.DOCUMENT_TEXT_DETECTION)],

            #image={"image": Image(content=image_content)},
            #features={"features": [Feature(type={"type": Feature.Type.TEXT_DETECTION})]},
        )

        response: AnnotateImageResponse = self.client.annotate_image(request=request)

        if not response.full_text_annotation:
            return None

        return response.full_text_annotation

