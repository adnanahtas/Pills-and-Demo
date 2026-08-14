module.exports = {
  apps: [
    {
      name: "qicscan",
      script: "venv/Scripts/python",
      args: "server.py",
      cwd: __dirname,
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        PORT: "8080",
        HOST: "0.0.0.0",
      },
    },
  ],
};
