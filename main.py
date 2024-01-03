import replicate
import streamlit as st
import requests
import zipfile
import io

# UI configurations
st.set_page_config(page_title="Replicate Image Generator", layout="wide")

st.title("Generator slika ✨")

# API Tokens and endpoints from `.streamlit/secrets.toml` file
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
REPLICATE_MODEL_ENDPOINTSTABILITY = st.secrets[
    "REPLICATE_MODEL_ENDPOINTSTABILITY"]

# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()


def configure_sidebar() -> None:
  """
    Setup and display the sidebar elements.

    This function configures the sidebar of the Streamlit application, 
    including the form for user inputs and the resources section.
    """
  with st.sidebar:
    with st.form("my_form"):
      with st.expander("**⚙️ PODEŠAVANJA**"):
        # Advanced Settings (for the curious minds!)
        width = st.number_input("Širina generisane slike", value=1024)
        height = st.number_input("Visina generisane slike", value=1024)
        num_outputs = st.slider("Broj generisanih slika",
                                value=1,
                                min_value=1,
                                max_value=4)
        scheduler = st.selectbox(
            'Scheduler komponenta',
            ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete', 'KarrasDPM',
             'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
        num_inference_steps = st.slider("Broj koraka denoising-a",
                                        value=50,
                                        min_value=1,
                                        max_value=500)
        guidance_scale = st.slider("Skala za klasifikator bez vođenja",
                                   value=7.5,
                                   min_value=1.0,
                                   max_value=50.0,
                                   step=0.1)
        prompt_strength = st.slider(
            "Prompt strength when using img2img/inpaint(1.0 corresponds to full destruction of infomation in image)",
            value=0.8,
            max_value=1.0,
            step=0.1)
        refine = st.selectbox(
            "Jačina podsticaja prilikom korišćenja img2img/inpaint",
            ("expert_ensemble_refiner", "None"))
        high_noise_frac = st.slider(
            "Izbor stila za preciziranje `expert_ensemble_refiner`",
            value=0.8,
            max_value=1.0,
            step=0.1)
      prompt = st.text_area(
          "**✨ Ukucaj prompt na engleskom**",
          value=
          "A glass of beer on the table in a night bar, cinematic, dramatic")
      negative_prompt = st.text_area(
          "**❌ Negativan prompt**",
          value="the absolute worst quality, distorted features",
          help=
          "Negativan prompt služi da programu zadate sve ono što ne želite da generiše"
      )

      # The Big Red "Submit" Button!
      submitted = st.form_submit_button("GENERIŠI SLIKU",
                                        type="primary",
                                        use_container_width=True)

    return submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt


def main_page(submitted: bool, width: int, height: int, num_outputs: int,
              scheduler: str, num_inference_steps: int, guidance_scale: float,
              prompt_strength: float, refine: str, high_noise_frac: float,
              prompt: str, negative_prompt: str) -> None:
  """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
        width (int): Width of the output image.
        height (int): Height of the output image.
        num_outputs (int): Number of images to output.
        scheduler (str): Scheduler type for the model.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Scale for classifier-free guidance.
        prompt_strength (float): Prompt strength when using img2img/inpaint.
        refine (str): Refine style to use.
        high_noise_frac (float): Fraction of noise to use for `expert_ensemble_refiner`.
        prompt (str): Text prompt for the image generation.
        negative_prompt (str): Text prompt for elements to avoid in the image.
    """
  if submitted:
    with st.status('✨ Slika se generiše..', expanded=True) as status:
      try:
        # Only call the API if the "Submit" button was pressed
        if submitted:
          # Calling the replicate API to get the image
          with generated_images_placeholder.container():
            all_images = []  # List to store all generated images
            output = replicate.run(REPLICATE_MODEL_ENDPOINTSTABILITY,
                                   input={
                                       "prompt": prompt,
                                       "width": width,
                                       "height": height,
                                       "num_outputs": num_outputs,
                                       "scheduler": scheduler,
                                       "num_inference_steps":
                                       num_inference_steps,
                                       "guidance_scale": guidance_scale,
                                       "prompt_stregth": prompt_strength,
                                       "refine": refine,
                                       "high_noise_frac": high_noise_frac
                                   })
            if output:
              st.toast('Tvoja slika je generisana!', icon='😍')
              # Save generated image to session state
              st.session_state.generated_image = output

              # Displaying the image
              for image in st.session_state.generated_image:
                with st.container():
                  st.image(image,
                           caption="Generisana slika ✨",
                           use_column_width=True)
                  # Add image to the list
                  all_images.append(image)

                  response = requests.get(image)
            # Save all generated images to session state
            st.session_state.all_images = all_images

            # Create a BytesIO object
            zip_io = io.BytesIO()

            # Download option for each image
            with zipfile.ZipFile(zip_io, 'w') as zipf:
              for i, image in enumerate(st.session_state.all_images):
                response = requests.get(image)
                if response.status_code == 200:
                  image_data = response.content
                  # Write each image to the zip file with a name
                  zipf.writestr(f"output_file_{i+1}.png", image_data)
                else:
                  st.error(
                      f"Failed to fetch image {i+1} from {image}. Error code: {response.status_code}",
                      icon="🚨")
            # Create a download button for the zip file
            st.download_button("**PREUZMI SLIKU**",
                               data=zip_io.getvalue(),
                               file_name="output_files.zip",
                               mime="application/zip",
                               use_container_width=True)
        status.update(label="Slika je generisana.",
                      state="complete",
                      expanded=False)
      except Exception as e:
        print(e)
        st.error(f'Encountered an error: {e}', icon="🚨")

  # If not submitted, chill here 🍹
  else:
    pass


def main():
  """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
  submitted, width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac, prompt, negative_prompt = configure_sidebar(
  )
  main_page(submitted, width, height, num_outputs, scheduler,
            num_inference_steps, guidance_scale, prompt_strength, refine,
            high_noise_frac, prompt, negative_prompt)


if __name__ == "__main__":
  main()
