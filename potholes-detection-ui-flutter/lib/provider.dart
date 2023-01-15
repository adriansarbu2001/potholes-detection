import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:io';

class PotholesDetectionProvider extends ChangeNotifier {
  // final String baseUri = "http://10.0.2.2:5000";
  final String baseUri = "http://78.96.179.191:5000";
  Image? image = null;
  bool loading = false;
  String? error = null;

  Future<void> postPhoto(File imageFile) async {
    image = null;
    error = null;
    loading = true;
    notifyListeners();
    try {
      var response = await http.post(
        Uri.parse("$baseUri/potholes-detection"),
        body: await imageFile.readAsBytes(),
      );

      image = Image.memory(response.bodyBytes);
      loading = false;
      notifyListeners();
    } catch (e) {
      error = e.toString();
      loading = false;
      notifyListeners();
    }
  }
}
