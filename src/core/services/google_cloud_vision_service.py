from google.oauth2 import service_account
from google.cloud.vision_v1 import ImageAnnotatorClient, AnnotateImageRequest, AnnotateImageResponse, TextAnnotation
from google.cloud.vision_v1.types import Image, Feature, Vertex
from google.protobuf.internal.wire_format import INT32_MAX, INT32_MIN

from src.core.config import settings
from src.core.utils.text_utils import clean_quoting, clean_punctuation, has_ended_abruptly


def calculate_page_ratios(page_width, page_height, bounding_box):
    """Obtains the x, y, width and height ratios of a bounding box for the page they occupy.

    Calculates the x, y, width and height ratios of a bounding box respective to the total
    width and height of a page. Assumes the bounding box is a box with 4 vertices.

        :param page_width: The page width.
        :type page_width: int

        :param page_height: The page height.
        :type page_height: int

        :param bounding_box: A list with 4 vertices.
        :type bounding_box: list[Vertex]

        :returns geometry: A dict with the mentioned ratios
        :rtype: dict
    """

    xs = [v.x for v in bounding_box]
    ys = [v.y for v in bounding_box]

    x = min(xs)
    y = min(ys)

    width = max(xs) - x
    height = max(ys) - y

    return {
        "x_ratio": round(x / page_width, 4),
        "y_ratio": round(y / page_height, 4),
        "width_ratio": round(width / page_width, 4),
        "height_ratio": round(height / page_height, 4),
    }


def create_bounding_box(min_x, min_y, max_x, max_y):
    return [
        Vertex(x=min_x, y=min_y),
        Vertex(x=max_x, y=min_y),
        Vertex(x=min_x, y=max_y),
        Vertex(x=max_x, y=max_y),
    ]


def get_text_paragraphs(full_text_annotation):
    """Obtains paragraphs from a given full text annotation.

    Parses the content of a full text annotation to obtain the paragraphs in the OCR extracted text.

        :param full_text_annotation: The full_text_annotation item from an AnnotateImageResponse.
        :type full_text_annotation: TextAnnotation

        :returns list: The list of extracted paragraphs. Can be empty.
        :rtype: list[dict]
    """
    paragraph_content = []
    lines = []

    for page in full_text_annotation.pages:
        for block_id, block in enumerate(page.blocks):
            for paragraph in block.paragraphs:

                words = []
                min_x = INT32_MAX
                max_x = INT32_MIN
                min_y = INT32_MAX
                max_y = INT32_MIN

                skip_iteration = False

                for i, word in enumerate(paragraph.words):
                    if not skip_iteration:
                        current_word = "".join(symbol.text for symbol in word.symbols)

                        for vertex in word.bounding_box.vertices:
                            min_x = min(min_x, vertex.x)
                            max_x = max(max_x, vertex.x)
                            min_y = min(min_y, vertex.y)
                            max_y = max(max_y, vertex.y)

                        last_symbol = word.symbols[-1]
                        break_type = (last_symbol.property.detected_break.type
                                      if last_symbol.property and last_symbol.property.detected_break
                                      else None)

                        # Manage break type if it's not just a space
                        if (break_type is not None and break_type not in [
                            TextAnnotation.DetectedBreak.BreakType.SPACE,
                            TextAnnotation.DetectedBreak.BreakType.SURE_SPACE
                        ] and i + 1 < len(paragraph.words)):
                            next_word = "".join(symbol.text for symbol in paragraph.words[i + 1].symbols)

                            # Join hyphenated word, with manual check in case Vision doesn't type correctly
                            if break_type == TextAnnotation.DetectedBreak.BreakType.HYPHEN\
                                    or current_word != "-" and current_word.endswith("-"):
                                skip_iteration = True
                                words.append(current_word[:-1] + next_word)

                            else:
                                words.append(current_word)

                                # Separate lines if the next_word doesn't continue the current line
                                if break_type in [
                                    TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
                                    TextAnnotation.DetectedBreak.BreakType.LINE_BREAK
                                ] and not has_ended_abruptly(current_word, next_word):
                                    # Append current line
                                    lines.append({
                                        "block_id": block_id,
                                        "text": clean_punctuation(" ".join(clean_quoting(words))),
                                        "geometry": calculate_page_ratios(
                                            page.width,
                                            page.height,
                                            create_bounding_box(min_x, min_y, max_x, max_y)
                                        )
                                    })

                                    # Reset parameters
                                    words = []
                                    min_x = INT32_MAX
                                    max_x = INT32_MIN
                                    min_y = INT32_MAX
                                    max_y = INT32_MIN
                        else:
                            words.append(current_word)
                    else:
                        skip_iteration = False

                lines.append({
                    "block_id": block_id,
                    "text": clean_punctuation(" ".join(clean_quoting(words))),
                    "geometry": calculate_page_ratios(
                        page.width,
                        page.height,
                        create_bounding_box(min_x, min_y, max_x, max_y)
                    )
                })

                text = " ".join(clean_quoting(words))

                paragraph_content.append({
                    "block_id": block_id,
                    "text": clean_punctuation(text),
                    "geometry": calculate_page_ratios(page.width, page.height, paragraph.bounding_box.vertices),
                })

    return lines


class VisionService:
    def __init__(self):
        self.client = ImageAnnotatorClient(
            credentials=service_account.Credentials.from_service_account_file(
                settings.google_cloud_vision_api_credentials
            ),
            client_options={"api_endpoint": settings.google_cloud_vision_api_endpoint   }
        )


    def detect_text(self, image_content):
        """Extracts text from a given image.

        Calls the associated Google Cloud Vision API endpoint to detect text in an image
        and extract it through OCR.

            :param image_content: The raw bytes of the image to process.
            :type image_content: bytes

            :returns full_text_annotation: The list of extracted text annotations (TextAnnotation class),
                or None if no text was extracted.
            :rtype: TextAnnotation
        """
        # Despite IDE expecting dicts, AnnotateImageRequest accepts protobuf format
        request = AnnotateImageRequest(
            image=Image(content=image_content),
            features=[Feature(type=Feature.Type.DOCUMENT_TEXT_DETECTION)],
        )

        response: AnnotateImageResponse = self.client.annotate_image(request=request)

        if not response.full_text_annotation:
            return None

        return response.full_text_annotation

