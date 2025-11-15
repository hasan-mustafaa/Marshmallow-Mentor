using System;

using Npgsql;

public class DatabaseConnecter
{
  

  static string connectionString = "Host=148.230.90.188;Username=testuser;Password=test123$;Database=gptwrapperdb";
d
  

  public static void async connectDB(){
    
  await using var dataSource = NpgsqlDataSource.Create(connectionString);

  }

}
