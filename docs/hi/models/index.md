---
comments: true
description: Ultralytics द्वारा समर्थित YOLO समूह, SAM, MobileSAM, FastSAM, YOLO-NAS, और RT-DETR मॉडल्स की विविधता का पता लगाएं। CLI और Python उपयोग के लिए उदाहरणों के साथ शुरू हो जाएं।
keywords: Ultralytics, दस्तावेज़ीकरण, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, मॉडल, वास्तुरचना, Python, CLI
---

# Ultralytics द्वारा समर्थित मॉडल्स

Ultralytics के मॉडल दस्तावेज़ीकरण में आपका स्वागत है! हम विशेष टास्क जैसे [ऑब्जेक्ट डिटेक्शन](../tasks/detect.md), [इंस्टेंस सेग्मेंटेशन](../tasks/segment.md), [छवि श्रेणीबद्धीकरण](../tasks/classify.md), [पोज़ संख्यानन](../tasks/pose.md), और [मल्टी-ऑब्जेक्ट ट्रैकिंग](../modes/track.md) के लिए विशेष रूप से तैयार किए गए मॉडल का समर्थन करते हैं। अगर आप अपनी मॉडल वास्तुरचना को Ultralytics में योगदान देना चाहते हैं, तो हमारे [योगदान गाइड](../../help/contributing.md) की जांच करें।

!!! Note

    🚧 हमारी बहुभाषी दस्तावेज़ीकरण वर्तमान में निर्माणाधीन है, और हम उसे सुधारने के लिए कठिनताओं पर काम कर रहे हैं। धन्यवाद आपकी सहानुभूति के लिए! 🙏

## प्रमुख मॉडल

यहां कुछ महत्वपूर्ण मॉडल हैं जिन्हें समर्थित किया जाता है:

1. **[YOLOv3](../../models/yolov3.md)**: YOLO मॉडल परिवार का तीसरा संस्करण, जो मूल रूप में Joseph Redmon द्वारा विकसित किया गया था, जिसे उसकी क्षमता के लिए जाना जाता है सही समय में ऑब्जेक्ट डिटेक्शन की।
2. **[YOLOv4](../../models/yolov4.md)**: YOLOv3 के लिए एक डार्कनेट जन्मित अपडेट, जिसे Alexey Bochkovskiy ने 2020 में जारी किया।
3. **[YOLOv5](../../models/yolov5.md)**: यूल्ट्रालिटिक्स द्वारा योगदान की एक सुधारी हुई YOLO वास्तुरचि। पिछले संस्करणों की तुलना में बेहतर प्रदर्शन और गति विपरीत संलग्नाता प्रदान करने का वादा।
4. **[YOLOv6](../../models/yolov6.md)**: 2022 में [मेटुआन](https://about.meituan.com/) द्वारा जारी किया गया, और कंपनी के कई स्वतंत्र वितरण रोबोट में प्रयोग होता है।
5. **[YOLOv7](../../models/yolov7.md)**: YOLOv4 के लेखकों द्वारा जारी किए गए नवीनतम YOLO मॉडल्स।
6. **[YOLOv8](../../models/yolov8.md)**: YOLO परिवार का नवीनतम संस्करण, जिसमें घटनाक्रम सेग्मेंटेशन, पोज़/कीपॉइंट्स अनुमान, और वर्गीकरण जैसी बढ़ी हुई क्षमताएं होती हैं।
7. **[सेगमेंट एनीथिंग मॉडल (SAM)](../../models/sam.md)**: मेटा का सेगमेंट एनीथिंग मॉडल (SAM)।
8. **[मोबाइल सेगमेंट एनीथिंग मॉडल (MobileSAM)](../../models/mobile-sam.md)**: मोबाइल एप्लिकेशनों के लिए MobileSAM, Kyung Hee विश्वविद्यालय द्वारा।
9. **[फ़ास्ट सेगमेंट एनीथिंग मॉडल (FastSAM)](../../models/fast-sam.md)**: दृश्य और वीडियो विश्लेषण समूह, स्वचालन विज्ञान संस्थान, चीन संगठन द्वारा FastSAM।
10. **[YOLO-NAS](../../models/yolo-nas.md)**: YOLO Neural Architecture Search (NAS) मॉडल्स।
11. **[रियलटाइम डिटेक्शन ट्रांसफॉर्मर (RT-DETR)](../../models/rtdetr.md)**: बाइडू का PaddlePaddle रियलटाइम डिटेक्शन ट्रांसफॉर्मर (RT-DETR) मॉडल।

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube वीडियो प्लेयर" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>देखें:</strong> थोड़ी सी कोड के कुछ पंक्तियों में Ultralytics YOLO मॉडल चलाएँ।
</p>

## प्रारंभ करना: उपयोग उदाहरण

!!! Example "उदाहरण"

    === "Python"

        PyTorch पूर्व-प्रशिक्षित `*.pt` मॉडल और विन्यास `*.yaml` फ़ाइलों को पायथन में योगदान करने के लिए `YOLO()`, `SAM()`, `NAS()` और `RTDETR()` कक्षाओं को पास करके मॉडल की एक नमूना उत्पन्न की जा सकती है:

        ```python
        from ultralytics import YOLO

        # एक COCO-पूर्व-प्रशिक्षित YOLOv8n मॉडल लोड करें
        model = YOLO('yolov8n.pt')

        # मॉडल जानकारी प्रदर्शित करें (वैकल्पिक)
        model.info()

        # COCO8 उदाहरण डेटासेट पर मॉडल 100 एपॉक्स के लिए प्रशिक्षित करें
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # YOLOv8n मॉडल के साथ 'bus.jpg' छवि पर इंफ़ेरेंस चलाएँ
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        मॉडलों को सीधे चलाने के लिए CLI कमांड प्रदान किए गए हैं:

        ```bash
        # एक COCO-पूर्व-प्रशिक्षित YOLOv8n मॉडल लोड करें और इसे COCO8 उदाहरण डेटासेट पर 100 एपॉक्स के लिए प्रशिक्षित करें
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # एक COCO-पूर्व-प्रशिक्षित YOLOv8n मॉडल लोड करें और 'bus.jpg' छवि पर इंफ़ेरेंस चलाएँ
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## नए मॉडल का योगदान देना

Ultralytics में अपना मॉडल योगदान करने में रुचि है? शानदार! हमें हमेशा अपने मॉडल पोर्टफोलियो का विस्तार करने के लिए खुले दिमाग से आगे बढ़ने की आवश्यकता है।

1. **यहाँ से रिपॉजिटरी फोर्क करें**: सबसे पहले, [Ultralytics GitHub रिपॉजिटरी](https://github.com/ultralytics/ultralytics) को फोर्क करके शुरू करें।

2. **अपने फोर्क को क्लोन करें**: अपने फोर्क को अपनी स्थानीय मशीन पर क्लोन करें और काम करने के लिए एक नया शाखा बनाएं।

3. **अपना मॉडल लागू करें**: अपना मॉडल उन्नत करें और Coding मानकों और दिशानिर्देशिकाओं का पालन करते हुए इसे जोड़ें। हमारे [योगदान गाइड](../../help/contributing.md) में उपलब्ध विवरणीय चरणों के लिए संपर्क करें।

4. **पूरी तरह से परीक्षण करें**: अपना मॉडल सम्पूर्ण रूप से औधोगिक रूप से परीक्षण करें, एकांत में और पाइपलाइन का हिस्सा के रूप में।

5. **एक पुल अनुरोध बनाएं**: अपने मॉडल से संतुष्ट होने के बाद, मुख्य रिपॉजिटरी के लिए एक पुल अनुरोध बनाएं जिसका समीक्षा की जाएगी।

6. **कोड समीक्षा और मर्ज**: समीक्षा के बाद, यदि आपका मॉडल हमारे मानदंडों को पूरा करता है, तो यह मुख्य रिपॉजिटरी में मर्ज हो जाएगा।

विस्तृत चरणों के लिए, हमारे [योगदान गाइड](../../help/contributing.md) पर देखें।