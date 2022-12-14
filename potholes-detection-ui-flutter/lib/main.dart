import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:potholes_detection/provider.dart';
import 'package:provider/provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Potholes detection',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ),
      home: MultiProvider(
        providers: [
          ChangeNotifierProvider(
              create: (context) => PotholesDetectionProvider()),
        ],
        child: const MyHomePage(title: 'Potholes detection'),
      ),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late PotholesDetectionProvider _provider;
  File? imageFile;

  /// Get from gallery
  _getFromGallery() async {
    PickedFile? pickedFile = await ImagePicker().getImage(
      source: ImageSource.gallery,
      maxWidth: 1800,
      maxHeight: 1800,
    );
    if (pickedFile != null) {
      imageFile = File(pickedFile.path);
      _provider.postPhoto(imageFile!);
    }
  }

  /// Get from Camera
  _getFromCamera() async {
    PickedFile? pickedFile = await ImagePicker().getImage(
      source: ImageSource.camera,
      maxWidth: 1800,
      maxHeight: 1800,
    );
    if (pickedFile != null) {
      imageFile = File(pickedFile.path);
      _provider.postPhoto(imageFile!);
    }
  }

  @override
  Widget build(BuildContext context) {
    _provider = Provider.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Container(
        child: imageFile == null
            ? Container(
                alignment: Alignment.center,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: <Widget>[
                    ElevatedButton(
                      onPressed: () {
                        _getFromGallery();
                      },
                      child: const Text("PICK FROM GALLERY"),
                    ),
                    Container(
                      height: 40.0,
                    ),
                    ElevatedButton(
                      onPressed: () {
                        _getFromCamera();
                      },
                      child: const Text("PICK FROM CAMERA"),
                    )
                  ],
                ),
              )
            : _provider.loading
                ? const Center(child: CircularProgressIndicator())
                : _provider.error != null
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(_provider.error.toString()),
                            const SizedBox(height: 12),
                            ElevatedButton(
                              onPressed: () {
                                setState(() {
                                  imageFile = null;
                                });
                              },
                              child: const Text("Choose another"),
                            ),
                          ],
                        ),
                      )
                    : SingleChildScrollView(
                        child: Column(
                          children: [
                            Center(
                              child: Stack(
                                children: [
                                  // Image.file(
                                  //   imageFile!,
                                  //   fit: BoxFit.cover,
                                  // ),
                                  Container(
                                    width: 400.0,
                                    height: 400.0,
                                    alignment: Alignment.center,
                                    decoration: BoxDecoration(
                                      image: DecorationImage(
                                        image: Image.file(imageFile!).image,
                                        fit: BoxFit.fill,
                                      ),
                                    ),
                                  ),
                                  Image(
                                    image: _provider.image!.image,
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 12),
                            ElevatedButton(
                              onPressed: () {
                                setState(() {
                                  imageFile = null;
                                });
                              },
                              child: const Text("Choose another"),
                            ),
                            const SizedBox(height: 12),
                          ],
                        ),
                      ),
      ),
    );
  }
}
