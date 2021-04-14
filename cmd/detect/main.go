package main

import (
	"flag"
	"fmt"
	"os"

	"edge-detection/detect/app"
)

func main() {
	conf := parseFlags()
	if err := app.Run(conf); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func parseFlags() *app.Config {
	conf := &app.Config{}
	flag.StringVar(&conf.InputFilepath, "in", "", markRequired("input image filepath"))
	flag.StringVar(&conf.OutputFilepath, "out", "", markRequired("output image filepath"))
	flag.Parse()
	if conf.InputFilepath == "" {
		usagef("input filepath not provided")
	}
	if conf.OutputFilepath == "" {
		usagef("output filepath not provided")
	}
	return conf
}

func markRequired(usage string) string {
	return fmt.Sprintf("%s (required)", usage)
}

func errorf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, format, args...)
	os.Exit(1)
}

func usagef(format string, args ...interface{}) {
	flag.Usage()
	errorf(format, args...)
}
