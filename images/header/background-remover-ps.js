/**
 * README, Can 12.04.24
 * 1. Put all your images into one folder. Make sure the directory contains only the images.
 * 2. Modify `sourceFolder`; Absolute path of the folder works fine for me.
 * 3. Modify `saveFolder`; Absolute path of the folder works fine for me.
 * 4. Open Photoshop.
 * 5. Go to File.
 * 6. Go to Scripts.
 * 7. Go to Browse.
 * 8. Select this file.
 * 
 * It should start to sequentially remove background of your images--fingers crossed.
 * 
 * PS There might be much efficient ways to do this, but not a PS expert here :-(
 */

var sourceFolder = Folder(
    "C:\\Users\\saipc\\OneDrive\\Desktop\\SAPO-DOX\\Studium\\TuWien\\GraphicsSeminarSS24\\images\\header\\orig"
);

if (sourceFolder != null) {
    var fileList = sourceFolder.getFiles();
    //comment the above line and uncomment the following line to filter specific file types. 
    // the script will not work if you have any non-image file in the src folder so try filtering files types
    // if the script fails.
    // var fileList = sourceFolder..getFiles(/\.(jpg|tif|psd|crw|cr2|nef|dcr|dc2|raw|heic)$/i);
}

for (var a = 0; a < fileList.length; a++) {
    app.open(fileList[a]);

    // Select subject
    var idautoCutout = stringIDToTypeID("autoCutout");
    var desc01 = new ActionDescriptor();
    var idsampleAllLayers = stringIDToTypeID("sampleAllLayers");
    desc01.putBoolean(idsampleAllLayers, false);
    try {
        executeAction(idautoCutout, desc01, DialogModes.NO);
    } catch (err) {}

    // Invert the selection
    app.activeDocument.selection.invert();

    // Create a color to be used with the fill command
    var colorRef = new SolidColor();
    colorRef.rgb.red = 255;
    colorRef.rgb.green = 255;
    colorRef.rgb.blue = 255;

    // Now apply fill to the current selection
    app.activeDocument.selection.fill(colorRef);

    //enter path for where you want the file saved
    var saveFolder = new Folder(
        "C:\\Users\\saipc\\OneDrive\\Desktop\\SAPO-DOX\\Studium\\TuWien\\GraphicsSeminarSS24\\images\\header\\background_removed"
    );

    var fileName = app.activeDocument.name.replace(/\.[^\.]+$/, "");
    saveJPG(
        new File(saveFolder + "/" + Date.now() + "_" + fileName + ".jpg"),
        12
    );
    app.activeDocument.close(SaveOptions.DONOTSAVECHANGES);
    //fileList[a].remove();
}

function saveJPG(saveFile, jpegQuality) {
    saveFile = saveFile instanceof File ? saveFile : new File(saveFile);

    jpegQuality = jpegQuality || 12;

    var jpgSaveOptions = new JPEGSaveOptions();
    jpgSaveOptions.embedColorProfile = true;
    jpgSaveOptions.formatOptions = FormatOptions.STANDARDBASELINE;
    jpgSaveOptions.matte = MatteType.NONE;
    jpgSaveOptions.quality = jpegQuality;
    activeDocument.saveAs(saveFile, jpgSaveOptions, true, Extension.LOWERCASE);
}
