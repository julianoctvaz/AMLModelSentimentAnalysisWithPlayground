//: A Cocoa based Playground to present user interface
import CreateML
import AppKit
import PlaygroundSupport
import Cocoa

//let nibFile = NSNib.Name("MyView")
//var topLevelObjects : NSArray?
//
//
//
//Bundle.main.loadNibNamed(nibFile, owner:nil, topLevelObjects: &topLevelObjects)
//let views = (topLevelObjects as! Array<Any>).filter { $0 is NSView }
//
//// Present the view in Playground
//PlaygroundPage.current.liveView = views[0] as! NSView

let jsonUrl = URL(fileURLWithPath: "/Users/julianoctvaz/Desktop/sentences.json")

let data = try MLDataTable(contentsOf: jsonUrl)

let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)

let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "label")

// Training accuracy as a percentage
let trainingAccuracy = (1.0 - sentimentClassifier.trainingMetrics.classificationError) * 100

// Validation accuracy as a percentage
let validationAccuracy = (1.0 - sentimentClassifier.validationMetrics.classificationError) * 100


//avaliando precisao
let evaluationMetrics = sentimentClassifier.evaluation(on: testingData, textColumn: "text", labelColumn: "label")

// Evaluation accuracy as a percentage
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

//print(sentimentClassifier.validationMetrics)

//let confusion = sentimentClassifier.validationMetrics.confusion


//// Filter for rows which contain mistakes.
//let errors = confusion[confusion["True Label"] != confusion["Predicted"]]
//let mostCommonError = errors.rows.max { row1, row2 in row1["Count", Int.self]! < row2["Count", Int.self]! }
//print(mostCommonError ?? "The confusion table is empty.")
//// ["Predicted" : "tech", "True Label" : "business", "Count" : 9]

//var precisionRecall = sentimentClassifier.validationMetrics.precisionRecall


//salvando o modelo
let metadata = MLModelMetadata(author: "Juliano Vaz",
                               shortDescription: "A model trained to classify several text sentiment, neutral, positive, and negative",
                               version: "1.0")

///try sentimentClassifier.write(to: URL(fileURLWithPath: "/Users/julianoctvaz/Desktop/AnRandomSentimentClassifier.mlmodel"),
        //                      metadata: metadata)

try sentimentClassifier.prediction(from: "Apple is a terrible company!")

try sentimentClassifier.prediction(from: "Apple is a bad company!")

try sentimentClassifier.prediction(from: "I just found the best restaurant ever")

try sentimentClassifier.prediction(from: "@Facebook is a terrible company!")

try sentimentClassifier.prediction(from: "@Facebook is a terrible company!")

try sentimentClassifier.prediction(from: "Fuck you, drivers!")

try sentimentClassifier.prediction(from: "i like cycle path")

try sentimentClassifier.prediction(from: "I love Pokemon!")

try sentimentClassifier.prediction(from: "Best Sausage Egg and Cheese I've had this year!")

try sentimentClassifier.prediction(from: "Bates tastes ok")

try sentimentClassifier.prediction(from: "I hate pizza!")

try sentimentClassifier.prediction(from: "Netflix is an average company")

try sentimentClassifier.prediction(from: "Justin Bieber sucks!")

try sentimentClassifier.prediction(from: "Sunsets are beautiful!")

try sentimentClassifier.prediction(from: "Good Morning, Juliano!")

try sentimentClassifier.prediction(from: "OK I understand Thank you.")

try sentimentClassifier.prediction(from: "Tell me about you friends")
