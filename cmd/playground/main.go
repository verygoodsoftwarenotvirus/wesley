package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"reflect"
	"runtime"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
	"github.com/traefik/yaegi/interp"
	"github.com/traefik/yaegi/stdlib"
)

const (
	berlinLat  = "52.520008"
	berlinLong = "13.405"
)

func LookupCityLatitude(cityName string) string {
	switch strings.TrimSpace(strings.ToLower(cityName)) {
	case "berlin":
		return berlinLat
	default:
		return "0.0"
	}
}

func LookupCityLongitude(cityName string) string {
	switch strings.TrimSpace(strings.ToLower(cityName)) {
	case "berlin":
		return berlinLong
	default:
		return "0.01"
	}
}

func LookupWeatherByCoordinate(lat, long string) string {
	if lat == berlinLat && long == berlinLong {
		return "bright and sunny weather out there"
	} else if lat == berlinLong && long == berlinLat {
		return "high chance of hurricanes in the area"
	}
	return "overcast and grey, with a chance of hail"
}

func main() {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	i := NewInquiry(client)

	i.AddFunctionToRepertoire(LookupCityLatitude, "returns the latitude of a given city")
	i.AddFunctionToRepertoire(LookupCityLongitude, "returns the longitude of a given city")
	i.AddFunctionToRepertoire(LookupWeatherByCoordinate, "returns the weather for a given latitude and longitude")

	ctx := context.Background()
	answer, err := i.Answer(ctx, "What is the weather like in Berlin right now?")
	if err != nil {
		panic(err)
	}

	println(answer)
}

func jsonSchemaTypeForGoType(t string) jsonschema.DataType {
	switch t {
	case "string":
		return jsonschema.String
	case "struct":
		return jsonschema.Object
	case "float32", "float64":
		return jsonschema.Number
	case "int", "int8", "int16", "int32", "int64":
		return jsonschema.Integer
	case "[]":
		return jsonschema.Array
	case "bool":
		return jsonschema.Boolean
	default:
		panic("unknown type")
	}
}

func openAIDefinitionForFunction(t any, description string) openai.FunctionDefinition {
	if k := reflect.ValueOf(t).Kind(); k != reflect.Func {
		panic("invalid input type")
	}

	funcName := runtime.FuncForPC(reflect.ValueOf(t).Pointer()).Name()
	funcNameParts := strings.Split(funcName, ".")
	if len(funcNameParts) > 1 {
		funcName = funcNameParts[1]
	}

	params := jsonschema.Definition{
		Type:       jsonschema.Object,
		Properties: map[string]jsonschema.Definition{},
		Required:   []string{},
	}
	x := openai.FunctionDefinition{
		Name:        funcName,
		Description: description,
	}

	typ := reflect.TypeOf(t)

	parameterCount := typ.NumIn()
	for i := 0; i < parameterCount; i++ {
		pn := fmt.Sprintf("%d", i)
		params.Properties[pn] = jsonschema.Definition{
			Type:        jsonSchemaTypeForGoType(typ.In(i).String()),
			Description: "",
		}
		params.Required = append(params.Required, pn)
	}

	x.Parameters = params

	return x
}

type availableInquiryFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	FunctionDef openai.FunctionDefinition
}

type Inquiry struct {
	messages           []openai.ChatCompletionMessage
	functionRepertoire []availableInquiryFunction
	openaiClient       *openai.Client
	replExports        map[string]reflect.Value
}

func NewInquiry(openaiClient *openai.Client) *Inquiry {
	return &Inquiry{
		openaiClient:       openaiClient,
		functionRepertoire: []availableInquiryFunction{},
		replExports:        map[string]reflect.Value{},
		messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "Only use the functions and parameters you have been provided with. Argument responses should take the strict form of a map of numeric keys to string values.",
			},
		},
	}
}

func (i *Inquiry) AddFunctionToRepertoire(f any, description string) {
	funcName := strings.Split(runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name(), ".")[1]

	i.replExports[funcName] = reflect.ValueOf(f)
	i.functionRepertoire = append(i.functionRepertoire, availableInquiryFunction{
		Name:        funcName,
		Description: description,
		FunctionDef: openAIDefinitionForFunction(f, "returns the weather for a given latitude and longitude"),
	})
}

func (i *Inquiry) Answer(ctx context.Context, question string) (string, error) {
	var answer string

	i.messages = []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "Only use the functions and parameters you have been provided with.",
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: question,
		},
	}

	for answer == "" {
		if t, ok := ctx.Deadline(); ok && t.Before(time.Now()) {
			return "", errors.New("context deadline exceeded")
		}

		latestAnswer, command, err := i.submitQuestion(ctx)
		if err != nil {
			return "", err
		}

		if latestAnswer != "" {
			answer = latestAnswer
		}

		if command != "" {
			repl := interp.New(interp.Options{})
			if err = repl.Use(stdlib.Symbols); err != nil {
				return "", err
			}

			if err = repl.Use(interp.Exports{"provided/provided": i.replExports}); err != nil {
				return "", err
			}

			if _, err = repl.Eval(`import "provided"`); err != nil {
				return "", err
			}

			script := fmt.Sprintf(`provided.%s`, command)

			var outcome reflect.Value
			outcome, err = repl.Eval(script)
			if err != nil {
				return "", err
			}

			i.messages = append(i.messages, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleFunction,
				Content: outcome.String(),
				Name:    strings.Split(command, "(")[0],
			})
		}
	}

	return answer, nil
}

func (i *Inquiry) submitQuestion(ctx context.Context) (answer string, functionCall string, err error) {
	log.Println("making request to Open AI")

	funcDefs := []openai.FunctionDefinition{}
	for _, x := range i.functionRepertoire {
		funcDefs = append(funcDefs, x.FunctionDef)
	}

	result, completionErr := i.openaiClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       openai.GPT3Dot5Turbo16K0613,
		Functions:   funcDefs,
		Temperature: 1.0,
		TopP:        1.0,
		Messages:    i.messages,
	})
	if completionErr != nil {
		return "", "", err
	}

	if len(result.Choices) > 0 {
		firstChoice := result.Choices[0]
		if firstChoice.FinishReason == "function_call" {
			var rawArgs map[int]string
			argsToUnmarshal := firstChoice.Message.FunctionCall.Arguments
			if unmarshalErr := json.Unmarshal([]byte(argsToUnmarshal), &rawArgs); unmarshalErr != nil {
				return "", "", unmarshalErr
			}

			args := make([]string, len(rawArgs))
			for k, v := range rawArgs {
				args[k] = fmt.Sprintf("%q", v)
			}

			script := fmt.Sprintf(`%s(%s)`, firstChoice.Message.FunctionCall.Name, strings.Join(args, ", "))

			return "", script, nil
		} else {
			return firstChoice.Message.Content, "", nil
		}
	}

	return "", "", errors.New("no choices returned")
}
