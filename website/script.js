const sliderInput = document.querySelector('.slider-input');
const sliderImage = document.querySelector('.slider-image');

sliderInput.addEventListener('input', () => {
  const imageIndex = sliderInput.value;

  console.log(imageIndex);

  const imageName = `output/deconvolution/clb2/clb2_deconvolved_${imageIndex}.png`;
  sliderImage.setAttribute('src', imageName);
});
