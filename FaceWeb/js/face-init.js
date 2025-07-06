// 面部捕捉模块初始化
try {
  import('./face-capture.js').then(module => {
    // 初始化面部捕捉
    const faceCapture = new module.FaceCapture('face-capture-container');
    console.log('✅ 面部捕捉模块加载成功');
  }).catch(error => {
    console.warn('⚠️ 面部捕捉模块加载失败，但不影响训练功能:', error);
    // 不阻塞主要功能，只是没有面部捕捉而已
  });
} catch (error) {
  console.warn('⚠️ 面部捕捉模块初始化失败:', error);
} 