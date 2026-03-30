const { execSync } = require('child_process');
try {
  console.log(execSync('xyzrender --help').toString());
} catch (e) {
  console.error(e.stdout.toString());
}
