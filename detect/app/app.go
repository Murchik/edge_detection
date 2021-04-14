package app

import (
	"edge-detection/detect/internal/sobel"
	"fmt"
	"image"
	"os"

	"github.com/disintegration/gift"
	"github.com/disintegration/imaging"
)

type Config struct {
	InputFilepath  string
	OutputFilepath string
}

func Run(conf *Config) error {
	im, err := readImage(conf.InputFilepath)
	if err != nil {
		return fmt.Errorf("can't read input image: %w", err)
	}
	bounds := im.Bounds()
	rgba := image.NewRGBA(bounds)
	gift.New().Draw(rgba, im)
	if err := sobel.Apply(rgba.Pix, bounds.Dx(), bounds.Dy()); err != nil {
		return fmt.Errorf("can't apply sobel method: %w", err)
	}
	if err := imaging.Save(rgba, conf.OutputFilepath); err != nil {
		return fmt.Errorf("can't save image: %w", err)
	}
	return nil
}

func readImage(filepath string) (image.Image, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("can't open file: %w", err)
	}
	defer file.Close()
	im, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("can't decode image: %w", err)
	}
	return im, nil
}
